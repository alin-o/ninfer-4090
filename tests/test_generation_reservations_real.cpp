#include "ninfer/engine.h"
#include "runtime/engine/context_transfer_test_gate.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

// Hold the real transfer stream until the independent consumer observes cancellation.
// Program must retain both images and settle CUDA before releasing the cancelled owner.
struct TransferDelay {
    std::atomic<bool> entered{false};
    std::atomic<bool> released{false};
    std::atomic<bool> timed_out{false};
    ninfer::runtime::testing::ContextTransferTestGate gate{
        this, [](void* context, cudaStream_t stream) noexcept {
            return cudaLaunchHostFunc(
                stream,
                [](void* value) {
                    auto& self = *static_cast<TransferDelay*>(value);
                    if (self.entered.exchange(true)) { return; }
                    const auto deadline =
                        std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (!self.released.load()) {
                        if (std::chrono::steady_clock::now() >= deadline) {
                            self.timed_out = true;
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                },
                context);
        }};

    ~TransferDelay() {
        released = true;
        ninfer::runtime::testing::clear_generation_transfer_gate();
    }
};

struct Sink : ninfer::OutputSink {
    unsigned starts = 0;
    std::string content;
    std::string reasoning;

    void start(ninfer::GenerationStart) override { ++starts; }

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override {
        (delta.channel == ninfer::OutputChannel::Content ? content : reasoning) += delta.text;
    }
};

void run(const char* artifact, std::uint32_t concurrency, bool host, int cancel_index = -1,
         bool shared = false, std::uint32_t quantum = 64, bool tight_host = false) {
    ninfer::EngineOptions options;
    options.artifact_path      = artifact;
    options.max_context        = concurrency == 8 ? 1024 : 512;
    options.kv_capacity        = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.max_concurrency    = concurrency;
    options.pending_timeout_ms = 120000;
    options.prefill_chunk      = 128;
    options.generation_reservation_tokens        = quantum;
    options.use_cuda_graph                       = concurrency != 2;
    options.context_cache.host_kv_capacity_bytes = host ? (tight_host ? 32ULL : 512ULL) << 20 : 0;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    ninfer::Engine engine(options);
    const auto tokens =
        engine.tokenize_text("Write the numbers from one to one thousand, separated by commas:");
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 400;
    request.execution.allow_prefix_reuse      = shared;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 42;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    const auto prepare                        = [&] {
        if (!shared) { return engine.prepare_tokens(tokens); }
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.messages                = {
            {.role  = ninfer::ChatRole::System,
                                                   .parts = {{.text =
                                           "You are a precise assistant. Write only the requested numeric list. "
                                                                                 "Separate consecutive integers with commas and spaces. Do not explain "
                                                                                 "the task. "
                                                                                 "Continue the list until the requested upper bound, without skipping "
                                                                                 "numbers."}}},
            {.role  = ninfer::ChatRole::User,
                                                   .parts = {
                 {.text = "Write the numbers from one to one thousand, separated by commas:"}}}};
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        input.context_cache.markers.push_back({.after_message_count = 1});
        return engine.prepare(std::move(input));
    };
    const auto baseline = engine.generate(prepare(), request);
    require(baseline.generated_token_ids.size() == 400, "baseline output budget");
    TransferDelay transfer_delay;
    const bool delayed_cancellation = cancel_index > 0;
    if (delayed_cancellation) {
        ninfer::runtime::testing::install_generation_transfer_gate(&transfer_delay.gate);
    }
    const auto before               = engine.runtime_stats();
    const auto cohort_started       = std::chrono::steady_clock::now();
    std::uint64_t peak_host_backing = 0;
    double max_queue_wait           = 0;
    double max_first_token          = 0;
    double max_completion           = 0;
    std::vector<ninfer::GenerationHandle> handles;
    for (std::uint32_t i = 0; i < concurrency; ++i) {
        handles.push_back(engine.submit(prepare(), request, ninfer::OutputConsumerMode::Streaming));
    }
    std::vector<std::future<ninfer::GenerationResult>> results;
    std::vector<Sink> sinks(concurrency);
    for (std::size_t i = 0; i < handles.size(); ++i) {
        results.push_back(std::async(std::launch::async, [&, i] {
            return handles[i].wait(&sinks[i], ninfer::CancellationView([&, i] {
                if (delayed_cancellation && static_cast<int>(i) == cancel_index) {
                    if (!transfer_delay.entered.load()) { return false; }
                    transfer_delay.released = true;
                    return true;
                }
                return static_cast<int>(i) == cancel_index &&
                       engine.runtime_stats().generation_suspensions >=
                           before.generation_suspensions +
                               (cancel_index == 0 ? 1 : concurrency - 1);
            }));
        }));
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
        while (results[i].wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
            peak_host_backing =
                std::max(peak_host_backing, engine.runtime_stats().generation_host_backing_bytes);
        }
        auto result     = results[i].get();
        max_queue_wait  = std::max(max_queue_wait, result.engine_timing.queue_wait_seconds);
        max_first_token = std::max(max_first_token, result.timings.first_token_seconds);
        max_completion  = std::max(max_completion, result.timings.total_seconds);
        require(sinks[i].starts == 1, "suspension restarted the response stream");
        require(sinks[i].content == result.content && sinks[i].reasoning == result.reasoning,
                "suspension duplicated or lost streamed output");
        if (static_cast<int>(i) == cancel_index) {
            require(result.finish_reason == ninfer::FinishReason::Cancelled,
                    "cancellation did not settle");
            continue;
        }
        require(result.generated_token_ids.size() == 400, "suspension changed output budget");
        require(result.finish_reason == ninfer::FinishReason::OutputLimit,
                "capacity mislabeled as stop");
        require(result.generated_token_ids == baseline.generated_token_ids,
                "deterministic continuation differs from unsuspended baseline");
    }
    const auto after = engine.runtime_stats();
    if (delayed_cancellation) {
        require(transfer_delay.entered && transfer_delay.released && !transfer_delay.timed_out,
                "cancellation did not progress independently of the delayed spill");
    }
    if (host && quantum == 64) {
        require(after.generation_extension_grants > before.generation_extension_grants,
                "no incremental growth exercised");
        if (!tight_host) {
            require(after.generation_suspensions > before.generation_suspensions,
                    "no suspension exercised under simultaneous growth");
        }
        require(after.generation_suspensions >= after.generation_resumes &&
                    after.generation_suspensions - after.generation_resumes <=
                        (cancel_index > 0 ? 1U : 0U),
                "not all surviving suspended requests resumed");
    } else {
        require(after.generation_suspensions == 0, "zero Host budget suspended a request");
    }
    if (shared) {
        require(after.reused_prompt_tokens > before.reused_prompt_tokens,
                "shared-prefix suspension fixture did not reuse a prefix");
    }
    require(after.generation_host_backing_bytes == 0, "live Host backing leaked after completion");
    const double cohort_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cohort_started).count();
    const auto memory = engine.memory_summary();
    std::cout << "quantum=" << quantum << " tight_host=" << tight_host
              << " concurrency=" << concurrency << " host=" << host
              << " extensions=" << after.generation_extension_grants
              << " suspensions=" << after.generation_suspensions
              << " resumes=" << after.generation_resumes
              << " transfer_bytes=" << after.generation_transfer_bytes << " shared=" << shared
              << " cancelled=" << cancel_index << " elapsed_s=" << cohort_seconds
              << " max_queue_s=" << max_queue_wait << " max_ttft_s=" << max_first_token
              << " max_completion_s=" << max_completion
              << " peak_host_backing=" << peak_host_backing << " device_pool_bytes="
              << memory.weights.capacity_bytes + memory.sequence.capacity_bytes +
                     memory.workspace.capacity_bytes
              << std::endl;
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (!artifact) {
        std::cout << "SKIP: Qwen3.8 artifact not configured\n";
        return 77;
    }
    try {
        for (const auto concurrency : {2U, 4U, 8U}) { run(artifact, concurrency, true); }
        run(artifact, 2, false);
        run(artifact, 4, true, 0);
        run(artifact, 4, true, 3);
        run(artifact, 4, true, -1, true);
        run(artifact, 2, true, -1, false, 64, true);
        run(artifact, 4, true, -1, false, 16384);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
