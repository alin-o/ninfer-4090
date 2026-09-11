#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/auto_save_writer.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/durable_shared_snapshot_access.h"
#include "runtime/engine/shared_snapshot_test_access.h"
#include "targets/registry.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
namespace {

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.auto_save_queue_jobs == 0) { options.auto_save_queue_jobs = 2; }
    if (options.auto_save_queue_bytes == 0) { options.auto_save_queue_bytes = 1ULL << 30U; }
    if (options.auto_save_queue_bytes == 0) {
        throw std::invalid_argument("auto-save queue byte capacity must be nonzero");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes               = cache.max_shared_prefixes.value_or(concurrency);
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

namespace {

std::string slot_model_binding(const LoadSummary& load) {
    return load.target + '\n' + load.model_id + '\n' + load.weights_id;
}

} // namespace

class Engine::Impl {
public:
    using Core27      = runtime::EngineCore<targets::Qwen3_6_27BInstance>;
    using Core35      = runtime::EngineCore<targets::Qwen3_6_35BA3BInstance>;
    using ScoreCore27 = runtime::CausalScoreCore<targets::Qwen3_6_27BInstance>;
    using ScoreCore35 = runtime::CausalScoreCore<targets::Qwen3_6_35BA3BInstance>;
    using Core = std::variant<std::monostate, std::unique_ptr<Core27>, std::unique_ptr<Core35>,
                              std::unique_ptr<ScoreCore27>, std::unique_ptr<ScoreCore35>>;

    explicit Impl(EngineOptions engine_options)
        : options(normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed  = targets::construct_target(options, device);
        active            = std::move(constructed.active);
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        core = std::visit(
            [&](auto& target_ptr) -> Core {
                using Instance =
                    typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore27>(*target_ptr, device);
                    }
                    return std::make_unique<Core27>(*target_ptr, device, options,
                                                    std::move(constructed.context_cost));
                } else {
                    if (options.purpose == EnginePurpose::CausalScoring) {
                        return std::make_unique<ScoreCore35>(*target_ptr, device);
                    }
                    return std::make_unique<Core35>(*target_ptr, device, options,
                                                    std::move(constructed.context_cost));
                }
            },
            active);
        if (options.auto_save_evicted) {
            auto_save_writer = std::make_unique<runtime::AutoSaveWriter>(
                options.auto_save_queue_jobs, options.auto_save_queue_bytes,
                options.auto_save_listener, write_snapshot_file);
            std::visit(
                [&](auto& constructed_core) {
                    if constexpr (requires {
                                      constructed_core->set_eviction_sink(
                                          std::string(),
                                          std::function<void(
                                              std::string,
                                              targets::qwen3_6::RetainedSessionSnapshot&&)>(),
                                          std::function<std::shared_ptr<void>(std::size_t)>());
                                  }) {
                        constructed_core->set_eviction_sink(
                            slot_model_binding(load),
                            [this](std::string path,
                                   targets::qwen3_6::RetainedSessionSnapshot&& snapshot) {
                                auto_save_writer->enqueue(std::move(path), std::move(snapshot));
                            },
                            [this](std::size_t bytes) { return auto_save_writer->reserve(bytes); });
                    }
                },
                core);
        }
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        // Auto-save snapshots retain producer events that reference the Program's device
        // context. Drain their bounded Host consumer before destroying the Engine core.
        if (auto_save_writer) { auto_save_writer->stop(); }
        core.emplace<std::monostate>();
        auto_save_writer.reset();
        try {
            device.synchronize();
        } catch (...) {}
    }

    // Blocks until every enqueued auto-save has been published. Explicit slot operations call
    // this before touching files so a pending write can never be read stale or interleave with
    // a client save of the same path.
    void drain_writes() {
        if (auto_save_writer) { auto_save_writer->drain(); }
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
    std::unique_ptr<runtime::AutoSaveWriter> auto_save_writer;

    [[nodiscard]] RuntimeStats with_auto_save_stats(RuntimeStats stats) const noexcept {
        return auto_save_writer ? auto_save_writer->populate_stats(std::move(stats)) : stats;
    }

public:
    static void write_snapshot_file(const std::string& path,
                                    const std::vector<std::uint8_t>& bytes);
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            if (token_ids.size() > target_ptr->capacity) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   context_capacity_error(token_ids.size(), target_ptr->capacity));
            }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.tokenize_text(text);
        },
        impl_->active);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore27>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore35>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.prompt_capabilities();
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore27>> ||
                                 std::is_same_v<CoreState, std::unique_ptr<Impl::ScoreCore35>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission =
                    core->submit(std::move(prompt.impl_->value), prompt_summary, prepare_seconds,
                                 std::move(resolved_options), consumer_mode, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode).wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.media_cache_summary();
        },
        impl_->active);
}

bool Engine::healthy() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) -> bool {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core->healthy();
            }
        },
        impl_->core);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->with_auto_save_stats(std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core));
}

// Write-then-rename keeps a torn write from ever shadowing a good snapshot at `path`. The
// staging name embeds the thread id so a concurrent auto-save of the same path never shares a
// temporary file.
void Engine::Impl::write_snapshot_file(const std::string& path,
                                       const std::vector<std::uint8_t>& bytes) {
    std::ostringstream staging_name;
    staging_name << path << ".tmp." << std::this_thread::get_id();
    const std::string staging = staging_name.str();
    {
        std::ofstream file(staging, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!file.good()) {
            file.close();
            (void)std::remove(staging.c_str());
            throw std::invalid_argument("failed to write session snapshot file");
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(staging, path, rename_error);
    if (rename_error) {
        (void)std::remove(staging.c_str());
        throw std::invalid_argument("failed to publish session snapshot file: " +
                                    rename_error.message());
    }
}

SlotSaveResult Engine::save_slot(std::uint32_t lane, const std::string& path,
                                 const std::string& expected_digest) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto started = std::chrono::steady_clock::now();
    // A pending auto-save of the same path must not land after this explicit save.
    impl_->drain_writes();
    const std::string binding                          = slot_model_binding(impl_->load);
    targets::qwen3_6::RetainedSessionSnapshot snapshot = std::visit(
        [&](auto& core) -> targets::qwen3_6::RetainedSessionSnapshot {
            if constexpr (requires {
                              core->save_retained_lane(lane, binding, expected_digest, path);
                          }) {
                return core->save_retained_lane(lane, binding, expected_digest, path);
            } else {
                throw std::logic_error("session persistence requires a generation Engine");
            }
        },
        impl_->core);

    Impl::write_snapshot_file(path, snapshot.bytes);
    if (impl_->auto_save_writer) {
        impl_->auto_save_writer->note_authoritative(path, snapshot.tokens);
    }

    SlotSaveResult result;
    result.tokens         = snapshot.tokens;
    result.bytes          = snapshot.bytes.size();
    result.session_digest = std::move(snapshot.session_digest);
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

SlotRestoreResult Engine::restore_slot(std::uint32_t lane, const std::string& path) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto started = std::chrono::steady_clock::now();
    // A restore must read the newest state, including a spill still in the writer queue.
    impl_->drain_writes();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { throw std::invalid_argument("session snapshot file is unavailable"); }
    const std::streamsize size = file.tellg();
    if (size <= 0) { throw std::invalid_argument("session snapshot file is empty"); }
    std::vector<std::uint8_t> snapshot(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(snapshot.data()), size);
    if (!file.good()) { throw std::invalid_argument("failed to read session snapshot file"); }
    file.close();

    const std::string binding = slot_model_binding(impl_->load);
    auto restored             = std::visit(
        [&](auto& core) -> std::pair<std::uint32_t, std::string> {
            const std::span<const std::uint8_t> bytes(snapshot.data(), snapshot.size());
            if constexpr (requires { core->restore_retained_lane(lane, bytes, binding, path); }) {
                return core->restore_retained_lane(lane, bytes, binding, path);
            } else {
                throw std::logic_error("session persistence requires a generation Engine");
            }
        },
        impl_->core);

    if (impl_->auto_save_writer) {
        impl_->auto_save_writer->note_authoritative(path, restored.first);
    }

    SlotRestoreResult result;
    result.tokens         = restored.first;
    result.bytes          = snapshot.size();
    result.session_digest = std::move(restored.second);
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

std::uint32_t Engine::erase_slot(std::uint32_t lane, const std::string& expected_digest) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](auto& core) -> std::uint32_t {
            if constexpr (requires { core->erase_retained_lane(lane, expected_digest); }) {
                return core->erase_retained_lane(lane, expected_digest);
            } else {
                throw std::logic_error("session persistence requires a generation Engine");
            }
        },
        impl_->core);
}

std::vector<SlotState> Engine::slot_states() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> std::vector<SlotState> {
            if constexpr (requires { core->slot_states(); }) {
                return core->slot_states();
            } else {
                throw std::logic_error("session persistence requires a generation Engine");
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

std::vector<runtime::DurableSharedSnapshotAccess::Candidate>
runtime::DurableSharedSnapshotAccess::candidates(Engine& engine, const PreparedPrompt& prompt) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    if (!prompt.impl_) { throw std::invalid_argument("PreparedPrompt is empty"); }
    return std::visit(
        [&](auto& core) -> std::vector<Candidate> {
            if constexpr (requires {
                              core->durable_shared_prefix_candidates(prompt.impl_->value);
                          }) {
                return core->durable_shared_prefix_candidates(prompt.impl_->value);
            } else {
                return {};
            }
        },
        engine.impl_->core);
}

runtime::DurableSharedSnapshotAccess::RecoveryDecision
runtime::DurableSharedSnapshotAccess::decide_recovery(
    Engine& engine, const PreparedPrompt& prompt, const RequestOptions& request_options,
    std::span<const Candidate> available_ssd_candidates) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    if (!prompt.impl_) { throw std::invalid_argument("PreparedPrompt is empty"); }
    RequestOptions copied                 = request_options;
    const ResolvedRequestOptions resolved = resolve_request_options(
        engine.impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(copied));
    return std::visit(
        [&](auto& core) -> RecoveryDecision {
            if constexpr (requires {
                              core->inspect_durable_shared_prefix_recovery(
                                  prompt.impl_->value, resolved.execution,
                                  available_ssd_candidates);
                          }) {
                const auto inspected = core->inspect_durable_shared_prefix_recovery(
                    prompt.impl_->value, resolved.execution, available_ssd_candidates);
                const Candidate* feasible_ssd = nullptr;
                std::size_t feasible_index    = 0;
                if (inspected.ssd_candidate_index &&
                    *inspected.ssd_candidate_index < available_ssd_candidates.size()) {
                    feasible_index = *inspected.ssd_candidate_index;
                    feasible_ssd   = &available_ssd_candidates[feasible_index];
                }
                if (inspected.warm_frontier != 0 &&
                    (feasible_ssd == nullptr ||
                     inspected.warm_frontier >= feasible_ssd->frontier)) {
                    const char* reason = "deeper-memory-ready";
                    if (feasible_ssd != nullptr &&
                        inspected.warm_frontier == feasible_ssd->frontier) {
                        reason = "same-boundary-memory-ready";
                    } else if (feasible_ssd == nullptr && !available_ssd_candidates.empty()) {
                        reason = "ssd-adoption-infeasible-memory-ready";
                    }
                    return {.source                   = RecoverySource::Memory,
                            .frontier                 = inspected.warm_frontier,
                            .estimated_memory_cost_ns = inspected.warm_cost_ns,
                            .reason                   = reason};
                }
                if (feasible_ssd != nullptr) {
                    return {.source    = RecoverySource::Ssd,
                            .candidate = *feasible_ssd,
                            .frontier  = feasible_ssd->frontier,
                            .reason    = feasible_index == 0 ? "ssd-deeper-feasible"
                                                             : "ssd-shallower-feasible"};
                }
                if (inspected.warm_frontier != 0) {
                    return {.source                   = RecoverySource::Memory,
                            .frontier                 = inspected.warm_frontier,
                            .estimated_memory_cost_ns = inspected.warm_cost_ns,
                            .reason                   = "ssd-adoption-infeasible-memory-ready"};
                }
                return {.reason = available_ssd_candidates.empty() ? "ssd-unavailable"
                                                                   : "ssd-adoption-infeasible"};
            }
            return {.reason = "ssd-engine-unsupported"};
        },
        engine.impl_->core);
}

runtime::DurableSharedSnapshotAccess::ImportResult
runtime::DurableSharedSnapshotAccess::import(Engine& engine, const Candidate& candidate,
                                             std::shared_ptr<const std::vector<std::uint8_t>> bytes,
                                             const CancellationView& cancellation) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    if (!bytes) { throw std::invalid_argument("durable shared snapshot payload is empty"); }
    if (cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "shared snapshot import was cancelled");
    }
    const std::string binding = slot_model_binding(engine.impl_->load);
    ImportResult imported     = std::visit(
        [&](auto& core) -> ImportResult {
            if constexpr (requires { core->shared_prefix_slot_summary(std::uint32_t{}); }) {
                std::uint64_t validation_nanoseconds = 0;
                std::uint64_t adoption_nanoseconds   = 0;
                bool validation_completed            = false;
                try {
                    const auto result = core->import_shared_prefix(
                        std::span<const std::uint8_t>(*bytes), binding, {}, &validation_nanoseconds,
                        &adoption_nanoseconds,
                        [&] {
                            if (cancellation.requested()) {
                                throw RequestError(RequestErrorKind::Cancelled,
                                                       "shared snapshot import was cancelled");
                            }
                        },
                        bytes, true, candidate, &validation_completed);
                    const auto summary = core->shared_prefix_slot_summary(result.slot);
                    if (!summary) {
                        throw std::logic_error("durable shared import has no catalogued summary");
                    }
                    return {
                            .disposition            = static_cast<std::uint32_t>(result.disposition),
                            .slot                   = result.slot,
                            .frontier               = summary->checkpoint.ref.frontier,
                            .validation_nanoseconds = validation_nanoseconds,
                            .adoption_nanoseconds   = adoption_nanoseconds,
                    };
                } catch (const RequestError&) {
                    throw;
                } catch (const std::invalid_argument& error) {
                    if (!validation_completed) { throw ValidationError(error.what()); }
                    throw;
                }
            } else {
                throw std::logic_error("durable shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
    if (cancellation.requested()) {
        // Adoption is atomic and remains a valid warm source. Cancellation only prevents the
        // caller from continuing to submission; it never tears down a committed shared owner.
        throw RequestError(RequestErrorKind::Cancelled, "shared snapshot import was cancelled");
    }
    return imported;
}

bool runtime::DurableSharedSnapshotAccess::resident(Engine& engine, const Candidate& candidate) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](auto& core) {
            if constexpr (requires { core->durable_shared_prefix_resident(candidate); }) {
                return core->durable_shared_prefix_resident(candidate);
            }
            return false;
        },
        engine.impl_->core);
}

bool runtime::DurableSharedSnapshotAccess::settle_export(Engine& engine, std::uint32_t slot,
                                                         std::uint64_t owner, bool committed) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](auto& core) {
            if constexpr (requires {
                              core->settle_durable_shared_prefix_export(slot, owner, committed);
                          }) {
                return core->settle_durable_shared_prefix_export(slot, owner, committed);
            }
            return false;
        },
        engine.impl_->core);
}

std::vector<runtime::DurableSharedSnapshotAccess::Export>
runtime::DurableSharedSnapshotAccess::begin_exports(
    Engine& engine, const std::function<std::shared_ptr<void>(std::size_t)>& reserve,
    const std::function<bool(std::uint32_t, std::uint64_t)>& claim,
    const std::function<void(std::uint32_t, std::uint64_t)>& relinquish) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::uint32_t capacity =
        engine.impl_->options.context_cache.max_shared_prefixes.value_or(0);
    const std::string binding = slot_model_binding(engine.impl_->load);
    std::vector<Export> exports;
    exports.reserve(capacity);
    for (std::uint32_t slot = 0; slot < capacity; ++slot) {
        std::optional<std::uint64_t> owner;
        try {
            owner = std::visit(
                [&](auto& core) -> std::optional<std::uint64_t> {
                    if constexpr (requires { core->durable_shared_prefix_owner(slot); }) {
                        return core->durable_shared_prefix_owner(slot);
                    }
                    return std::nullopt;
                },
                engine.impl_->core);
            if (!owner || (claim && !claim(slot, *owner))) { continue; }
            Snapshot snapshot = std::visit(
                [&](auto& core) -> Snapshot {
                    if constexpr (requires {
                                      core->begin_export_shared_prefix(slot, binding, reserve,
                                                                       *owner);
                                  }) {
                        return core->begin_export_shared_prefix(slot, binding, reserve, *owner);
                    } else {
                        return {};
                    }
                },
                engine.impl_->core);
            if (snapshot.queue_reservation) {
                exports.push_back(Export{
                    .snapshot = std::move(snapshot),
                    .slot     = slot,
                    .owner    = *owner,
                });
            } else if (relinquish) {
                relinquish(slot, *owner);
            }
        } catch (const RequestError& error) {
            if (owner && relinquish) { relinquish(slot, *owner); }
            if (error.kind() != RequestErrorKind::Overloaded) { throw; }
        } catch (const std::invalid_argument&) {
            // Vacant and non-durable shared entries are ordinary bounded-catalog misses.
            if (owner && relinquish) { relinquish(slot, *owner); }
        } catch (...) {
            if (owner && relinquish) { relinquish(slot, *owner); }
            throw;
        }
    }
    return exports;
}

std::pair<std::uint32_t, targets::qwen3_6::RetainedSessionSnapshot>
runtime::testing::SharedSnapshotTestAccess::export_first_durable(Engine& engine) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::uint32_t capacity =
        engine.impl_->options.context_cache.max_shared_prefixes.value_or(0);
    std::string rejections;
    for (std::uint32_t slot = 0; slot < capacity; ++slot) {
        try {
            return {slot, export_slot(engine, slot)};
        } catch (const std::invalid_argument& error) {
            if (!rejections.empty()) { rejections += "; "; }
            rejections += "slot " + std::to_string(slot) + ": " + error.what();
        }
    }
    throw std::invalid_argument("Engine has no durable shared prefix" +
                                (rejections.empty() ? std::string{} : ": " + rejections));
}

targets::qwen3_6::RetainedSessionSnapshot
runtime::testing::SharedSnapshotTestAccess::export_slot(Engine& engine, std::uint32_t slot) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::string binding = slot_model_binding(engine.impl_->load);
    return std::visit(
        [&](auto& core) -> targets::qwen3_6::RetainedSessionSnapshot {
            if constexpr (requires { core->begin_export_shared_prefix(slot, binding); }) {
                return core->begin_export_shared_prefix(slot, binding);
            } else {
                throw std::logic_error("shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
}

runtime::testing::SharedSnapshotImportObservation
runtime::testing::SharedSnapshotTestAccess::import(Engine& engine,
                                                   std::span<const std::uint8_t> bytes) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::string binding = slot_model_binding(engine.impl_->load);
    return std::visit(
        [&](auto& core) -> SharedSnapshotImportObservation {
            if constexpr (requires {
                              core->import_shared_prefix(bytes, binding);
                              core->shared_prefix_slot_summary(std::uint32_t{});
                          }) {
                const auto result  = core->import_shared_prefix(bytes, binding);
                const auto summary = core->shared_prefix_slot_summary(result.slot);
                if (!summary) {
                    throw std::logic_error("shared import result has no catalogued summary");
                }
                return {.disposition      = static_cast<std::uint32_t>(result.disposition),
                        .slot             = result.slot,
                        .frontier         = summary->checkpoint.ref.frontier,
                        .main_frontier    = summary->checkpoint.required_kv.main_frontier,
                        .backend_frontier = summary->checkpoint.required_kv.backend_frontier,
                        .state_residency  = summary->checkpoint.state_residency};
            } else {
                throw std::logic_error("shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
}

std::uint32_t
runtime::testing::SharedSnapshotTestAccess::import_cancelled(Engine& engine,
                                                             std::span<const std::uint8_t> bytes) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::string binding = slot_model_binding(engine.impl_->load);
    std::atomic<bool> cancelled{true};
    return std::visit(
        [&](auto& core) -> std::uint32_t {
            if constexpr (requires { core->import_shared_prefix(bytes, binding); }) {
                try {
                    const auto result = core->import_shared_prefix(
                        bytes, binding, runtime::CancellationFlagView{.flag = &cancelled});
                    return static_cast<std::uint32_t>(result.disposition);
                } catch (const RequestError& error) {
                    if (error.kind() == RequestErrorKind::Cancelled) { return 2U; }
                    throw;
                }
            } else {
                throw std::logic_error("shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
}

runtime::testing::SealedSharedSnapshotTestImport
runtime::testing::SharedSnapshotTestAccess::parse(Engine& engine,
                                                  std::span<const std::uint8_t> bytes) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::string binding = slot_model_binding(engine.impl_->load);
    return std::visit(
        [&](auto& core) -> SealedSharedSnapshotTestImport {
            if constexpr (requires { core->parse_shared_prefix_for_test(bytes, binding); }) {
                auto parsed  = core->parse_shared_prefix_for_test(bytes, binding);
                using Import = std::remove_cvref_t<decltype(parsed)>;
                SealedSharedSnapshotTestImport out;
                out.storage_ = std::make_shared<Import>(std::move(parsed));
                out.type_    = &typeid(Import);
                return out;
            } else {
                throw std::logic_error("shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
}

void runtime::testing::SharedSnapshotTestAccess::import_validated(
    Engine& engine, const SealedSharedSnapshotTestImport& imported) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    if (!imported.storage_ || imported.type_ == nullptr) {
        throw std::invalid_argument("sealed shared snapshot test import is empty");
    }
    std::visit(
        [&](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                using Core = typename CoreState::element_type;
                if constexpr (requires { typename Core::ValidatedSharedPrefixImport; }) {
                    using Import = typename Core::ValidatedSharedPrefixImport;
                    if (*imported.type_ != typeid(Import)) {
                        throw std::invalid_argument("sealed shared snapshot target family differs");
                    }
                    (void)core->adopt_validated_shared_prefix_for_test(
                        *std::static_pointer_cast<Import>(imported.storage_));
                } else {
                    throw std::logic_error("shared snapshots require a generation Engine");
                }
            } else {
                throw std::logic_error("shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
}

std::uint32_t runtime::testing::SharedSnapshotTestAccess::import_with_cancellation(
    Engine& engine, std::span<const std::uint8_t> bytes, std::atomic<bool>& cancellation) {
    if (!engine.impl_) { throw std::logic_error("Engine is moved from"); }
    const std::string binding = slot_model_binding(engine.impl_->load);
    return std::visit(
        [&](auto& core) -> std::uint32_t {
            if constexpr (requires { core->import_shared_prefix(bytes, binding); }) {
                const auto result = core->import_shared_prefix(
                    bytes, binding, runtime::CancellationFlagView{.flag = &cancellation});
                return static_cast<std::uint32_t>(result.disposition);
            } else {
                throw std::logic_error("shared snapshots require a generation Engine");
            }
        },
        engine.impl_->core);
}

} // namespace ninfer
