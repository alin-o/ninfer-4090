#include "serve/generation_service.h"
#include "serve/anthropic_messages.h"
#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/request_events.h"
#include "serve/request_log.h"
#include "runtime/engine/durable_shared_snapshot_access.h"
#include "runtime/engine/context_transfer_test_gate.h"
#include "runtime/engine/shared_snapshot_test_access.h"

#include <ninfer/build_identity.h>

#include <cuda_runtime.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace ninfer::serve::testing {

struct GenerationServiceTestAccess {
    static Engine& engine(GenerationService& service) { return *service.engine_; }
    static void set_before_payload_read(GenerationService& service,
                                        std::function<void()> callback) {
        service.set_before_payload_read_for_test(std::move(callback));
    }
};

} // namespace ninfer::serve::testing

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::serve;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

RuntimeStats settled_stats(const GenerationService& service, bool wait_for_capture = false) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto stats = service.runtime_stats();
        if (stats.running_requests == 0 && stats.waiting_requests == 0 &&
            stats.materializing_requests == 0 &&
            (!wait_for_capture || stats.capture_pending_requests == 0)) {
            return stats;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("OpenAI cache request did not settle");
}

ServeOptions options(const char* artifact) {
    ServeOptions result;
    result.artifact_path                           = artifact;
    result.max_context                             = 4096;
    result.kv_capacity                             = KvCapacityPolicy::explicit_capacity(4096);
    result.max_concurrency                         = 1;
    result.pending_timeout_ms                      = 120000;
    result.prefill_chunk                           = 1024;
    result.kv_cache                                = KvCacheStorage::RK4V4E8;
    result.speculative.backend                     = SpeculativeBackend::Mtp;
    result.speculative.draft_tokens                = 3;
    result.speculative.proposal_head               = ProposalHead::Optimized;
    result.context_cache.device_state_slots        = 1;
    result.context_cache.host_state_slots          = 16;
    result.context_cache.host_kv_capacity_bytes    = 1ULL << 30;
    result.context_cache.max_private_continuations = 4;
    result.context_cache.max_shared_prefixes       = 8;
    result.context_cache.max_long_anchors_per_continuation = 0;
    result.enable_thinking                                 = false;
    result.preserve_thinking                               = true;
    result.greedy                                          = true;
    result.sampling_overrides.seed                         = 42;
    return result;
}

GenerationOutcome generate_history(GenerationService& service, const Json& history, bool responses,
                                   bool explicit_only                     = false,
                                   std::optional<std::string> session_key = std::nullopt) {
    Json body{{"model", "qwen3.8"}};
    if (explicit_only) { body["prompt_cache_options"] = Json{{"mode", "explicit"}}; }
    GenerationRequest generation;
    ContextCacheHints hints;
    if (responses) {
        body["store"]             = false;
        body["input"]             = history;
        body["max_output_tokens"] = 8;
        const auto request        = parse_openai_responses_create_request(body, RequestLimits{});
        OpenAIResponsesStore store(8, 1ULL << 20);
        auto resolved =
            resolve_openai_responses_prompt(request.prompt, store, "resp_cache", request.store);
        generation = std::move(resolved.generation);
        hints      = std::move(resolved.cache_hints);
    } else {
        body["messages"]   = history;
        body["max_tokens"] = 8;
        generation         = parse_chat_completion_request(body, RequestLimits{}).generation;
    }
    if (session_key) { hints.session_key = std::move(*session_key); }
    // Exercise the production merge of protocol markers/policy with Responses retention hints.
    auto prepared = service.prepare(generation, GenerationConsumerMode::Aggregate, {}, hints);
    auto result   = service.run(prepared, nullptr);
    (void)settled_stats(service);
    require(!result.generated_token_ids.empty(), "OpenAI cache fixture generated no tokens");
    return result;
}

GenerationOutcome generate(GenerationService& service, const std::string& instructions,
                           const std::string& user, bool responses, bool explicit_only = false,
                           std::optional<std::string> session_key = std::nullopt) {
    return generate_history(service,
                            Json::array({Json{{"role", "system"}, {"content", instructions}},
                                         Json{{"role", "user"}, {"content", user}}}),
                            responses, explicit_only, std::move(session_key));
}

void exercise_disconnect_after_checkpoint_reuse(const char* artifact) {
    GenerationService service(options(artifact));
    const Json history =
        Json::array({Json{{"role", "user"}, {"content", "Retain this exact disconnect fixture."}}});
    const GenerationOutcome seeded = generate_history(service, history, false);
    require(seeded.checkpoints.created_committed != 0,
            "disconnect fixture did not create a reusable checkpoint");

    Json body{{"model", "qwen3.8"}, {"messages", history}, {"max_tokens", 8}, {"stream", true}};
    GenerationRequest request = parse_chat_completion_request(body, RequestLimits{}).generation;
    PreparedRequest prepared  = service.prepare(request, GenerationConsumerMode::Streaming);
    StreamSink sink;
    sink.on_content   = [](const std::string&) { throw ClientDisconnected(); };
    sink.on_reasoning = [](const std::string&) { throw ClientDisconnected(); };
    bool disconnected = false;
    try {
        (void)service.run(prepared, &sink);
    } catch (const ClientDisconnected&) { disconnected = true; }

    const auto loaded =
        std::find_if(prepared.failure_checkpoint_lifecycle.begin(),
                     prepared.failure_checkpoint_lifecycle.end(), [](const auto& fact) {
                         return (fact.operation == CheckpointLifecycleOperation::Loaded ||
                                 fact.operation == CheckpointLifecycleOperation::Restored) &&
                                fact.status == CheckpointLifecycleStatus::Committed &&
                                fact.frontier != 0 &&
                                fact.key_digests != std::array<std::uint64_t, 2>{};
                     });
    const RequestFailure failure = attach_checkpoint_lifecycle(
        make_client_disconnected_failure(RequestFailurePhase::Transport),
        prepared.failure_checkpoint_lifecycle);
    require(disconnected && loaded != prepared.failure_checkpoint_lifecycle.end() &&
                failure.classification == RequestFailureClass::ClientDisconnected &&
                summarize_checkpoint_lifecycle(failure.checkpoint_lifecycle).restore_committed,
            "disconnect after reuse lost settled facts or changed transport classification");
}

void exercise_stream_failure_before_wait(const char* artifact) {
    GenerationService service(options(artifact));
    const auto prepare = [&] {
        Json body{{"model", "qwen3.8"},
                  {"messages", Json::array({Json{{"role", "user"},
                                                 {"content", "cancel before stream start"}}})},
                  {"max_tokens", 8},
                  {"stream", true}};
        GenerationRequest request = parse_chat_completion_request(body, RequestLimits{}).generation;
        PreparedRequest prepared  = service.prepare(request, GenerationConsumerMode::Streaming);
        // SSD adoption is synchronous in prepare(). Model that already-committed fact explicitly
        // so both gateway paths prove they merge it while cancelling the submitted Engine work.
        prepared.durable_lifecycle.push_back(CheckpointLifecycleFact{
            .key_digests      = {0x1234, 0x5678},
            .content_digest   = std::string(64, 'a'),
            .frontier         = 64,
            .role             = CheckpointLifecycleRole::SharedStablePrefix,
            .scope            = CheckpointLifecycleScope::Shared,
            .operation        = CheckpointLifecycleOperation::Restored,
            .source_tier      = CheckpointLifecycleTier::Ssd,
            .destination_tier = CheckpointLifecycleTier::Device,
            .status           = CheckpointLifecycleStatus::Committed,
            .state_images     = 1,
            .main_kv_pages    = 1,
        });
        return prepared;
    };
    const auto verify = [&](PreparedRequest prepared, bool fail_initial_write, const char* label) {
        if (fail_initial_write) {
            httplib::DataSink failed_sink;
            failed_sink.write = [](const char*, std::size_t) { return false; };
            std::atomic<bool> cancelled{false};
            SseTransport transport(failed_sink, cancelled);
            bool disconnected = false;
            try {
                transport.write("data: initial-event\n\n");
            } catch (const ClientDisconnected&) { disconnected = true; }
            require(disconnected && cancelled.load(std::memory_order_acquire),
                    "initial SSE write fixture did not fail at the transport boundary");
        }
        const std::vector<CheckpointLifecycleFact> facts = service.cancel_and_settle(prepared);
        const auto restored = std::find_if(facts.begin(), facts.end(), [](const auto& fact) {
            return fact.operation == CheckpointLifecycleOperation::Restored &&
                   fact.source_tier == CheckpointLifecycleTier::Ssd &&
                   fact.status == CheckpointLifecycleStatus::Committed;
        });
        const RequestFailure failure = attach_checkpoint_lifecycle(
            make_client_disconnected_failure(RequestFailurePhase::Transport), facts);
        require(restored != facts.end() &&
                    summarize_checkpoint_lifecycle(facts).restore_committed &&
                    failure.classification == RequestFailureClass::ClientDisconnected &&
                    failure.phase == RequestFailurePhase::Transport,
                label);
        (void)settled_stats(service);
    };

    verify(prepare(), true, "initial SSE write failure lost pre-generation checkpoint facts");
    verify(prepare(), false,
           "stream provider that never started lost pre-generation checkpoint facts");
}

void exercise_harness(const char* artifact) {
    std::string harness;
    for (unsigned index = 0; index < 1500; ++index) { harness += "stable "; }
    harness += "\n=== CACHE_BREAKPOINT ===\n";
    const auto instructions = [&](const std::string& workspace) {
        return harness + "Current working directory: " + workspace;
    };
    GenerationOutcome restored;
    GenerationOutcome continued;
    Json continuation;
    {
        GenerationService service(options(artifact));
        // Explicit-only with no markers must not seed any automatic shared writes.
        (void)generate(service, instructions("explicit"), "Reply yes.", true, true);
        const auto seed = generate(service, instructions("first"), "Reply yes.", true);
        require(seed.metrics.prefix_cache_hit_tokens == 0,
                "explicit-only request unexpectedly published an automatic harness");
        const auto before = settled_stats(service);
        // Both the volatile instruction suffix and the conversation change. Private endpoints,
        // full instructions and last-content markers cannot match this request.
        restored         = generate(service, instructions("second"), "Reply no.", false);
        const auto after = settled_stats(service);
        std::cout << "openai_harness prompt_tokens=" << restored.prompt_tokens
                  << " reused_tokens=" << restored.metrics.prefix_cache_hit_tokens
                  << " path=" << static_cast<int>(restored.metrics.prefix_reuse_path)
                  << " state_d2h=" << before.state_d2h_count
                  << " state_h2d=" << after.state_h2d_count - before.state_h2d_count
                  << " main_h2d_pages=" << after.main_kv_h2d_pages - before.main_kv_h2d_pages
                  << " mtp_h2d_pages=" << after.backend_kv_h2d_pages - before.backend_kv_h2d_pages
                  << '\n';
        require(restored.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
                    restored.metrics.prefix_cache_hit_tokens >= 1500,
                "changed OpenAI conversation missed its shared harness");
        require(after.state_h2d_count > before.state_h2d_count,
                "shared harness regression did not exercise RAM State reload");
        // Explicit-only remains able to read an existing exact shared prefix.
        const auto explicit_read =
            generate(service, instructions("third"), "Reply yes.", true, true);
        require(explicit_read.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
                    explicit_read.metrics.prefix_cache_hit_tokens ==
                        restored.metrics.prefix_cache_hit_tokens,
                "explicit-only request could not read the existing harness");
        const auto changed = generate(service,
                                      "A different harness.\n=== CACHE_BREAKPOINT ===\n"
                                      "Current working directory: second",
                                      "Reply no.", true);
        require(changed.metrics.prefix_cache_hit_tokens == 0,
                "changed harness incorrectly reused an unrelated prefix");
        require(restored.finish_reason == FinishReason::StopToken,
                "continuation fixture must include the completed assistant response");
        continuation                = Json::array({
            Json{{"role", "system"}, {"content", instructions("second")}},
            Json{{"role", "user"}, {"content", "Reply no."}},
            Json{{"role", "assistant"}, {"content", restored.text}},
            Json{{"role", "user"}, {"content", "Continue briefly."}},
        });
        const auto before_continue  = settled_stats(service);
        continued                   = generate_history(service, continuation, true);
        const auto after_continue   = settled_stats(service);
        const auto completed_tokens = restored.prompt_tokens + restored.completion_tokens;
        std::cout << "openai_continuation completed_tokens=" << completed_tokens
                  << " prompt_tokens=" << continued.prompt_tokens
                  << " reused_tokens=" << continued.metrics.prefix_cache_hit_tokens
                  << " path=" << static_cast<int>(continued.metrics.prefix_reuse_path)
                  << " state_h2d="
                  << after_continue.state_h2d_count - before_continue.state_h2d_count << '\n';
        require(continued.metrics.prefix_reuse_path == PrefixReusePath::PrivateEndpoint &&
                    continued.metrics.prefix_cache_hit_tokens >= completed_tokens - 1,
                "Responses continuation lost the completed assistant response");
    }
    auto cold_options               = options(artifact);
    cold_options.allow_prefix_reuse = false;
    GenerationService cold_service(cold_options);
    const auto cold = generate(cold_service, instructions("second"), "Reply no.", false);
    require(cold.metrics.prefix_cache_hit_tokens == 0 &&
                cold.generated_token_ids == restored.generated_token_ids &&
                cold.metrics.speculative_draft_tokens ==
                    restored.metrics.speculative_draft_tokens &&
                cold.metrics.speculative_accepted_tokens ==
                    restored.metrics.speculative_accepted_tokens,
            "RAM harness reuse diverged from cold tokens or MTP counters");
    std::cout << "openai_harness exact_cold_match=true\n";
    const auto cold_continuation = generate_history(cold_service, continuation, true);
    require(cold_continuation.metrics.prefix_cache_hit_tokens == 0 &&
                cold_continuation.generated_token_ids == continued.generated_token_ids &&
                cold_continuation.metrics.speculative_draft_tokens ==
                    continued.metrics.speculative_draft_tokens &&
                cold_continuation.metrics.speculative_accepted_tokens ==
                    continued.metrics.speculative_accepted_tokens,
            "Responses continuation diverged from cold tokens or MTP counters");
    std::cout << "openai_continuation exact_cold_match=true\n";
}

struct TemporaryDirectory {
    TemporaryDirectory() {
        std::string pattern = "/tmp/ninfer-request-capture-real-XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const char* created = ::mkdtemp(writable.data());
        if (created == nullptr) { throw std::runtime_error("capture mkdtemp failed"); }
        path = created;
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)), {});
}

void exercise_generated_reasoning_continuation(const char* artifact) {
    TemporaryDirectory directory;
    auto configured = options(artifact);
    // This checks reasoning identity and persistence. Leave room to fork the preserved source
    // and its rewrite State; a two-image pool can legitimately favor root over destroying the
    // anonymous source's recovery point once consumption has a retention cost.
    configured.context_cache.device_state_slots = 3;
    // The second replay must use the restored private endpoint, not a shared checkpoint
    // published while running the first replay.
    configured.context_cache.max_shared_prefixes = 0;
    GenerationService service(configured);
    const auto run = [&](const Json& history, int max_tokens) {
        const Json body{{"model", "qwen3.8"}, {"messages", history}, {"max_tokens", max_tokens}};
        auto request              = parse_chat_completion_request(body, RequestLimits{}).generation;
        request.enable_thinking   = true;
        request.preserve_thinking = true;
        auto prepared             = service.prepare(request, GenerationConsumerMode::Aggregate);
        auto result               = service.run(prepared, nullptr);
        (void)settled_stats(service, true);
        return result;
    };
    Json history =
        Json::array({Json{{"role", "user"}, {"content", "Hi. Reply with a short greeting."}}});
    const auto source = run(history, 512);
    require(source.finish_reason == FinishReason::StopToken && !source.reasoning.empty() &&
                !source.text.empty() && source.id_slot >= 0,
            "reasoning continuation fixture did not retain a completed thinking response");
    const auto snapshot = directory.path / "reasoning.nss";
    (void)service.slot_save(source.id_slot, snapshot.string(), source.session_digest);
    history.push_back(Json{
        {"role", "assistant"}, {"content", source.text}, {"reasoning_content", source.reasoning}});
    history.push_back(Json{{"role", "user"}, {"content", "Say hello once more."}});
    const auto warm = run(history, 8);
    const auto endpoint =
        static_cast<std::uint32_t>(source.prompt_tokens + source.completion_tokens - 1);
    std::cout << "reasoning_warm endpoint=" << endpoint
              << " reused=" << warm.metrics.prefix_cache_hit_tokens
              << " path=" << static_cast<unsigned>(warm.metrics.prefix_reuse_path)
              << " now_ns=" << warm.metrics.materialization.predicted_now_ns
              << " loss_ns=" << warm.metrics.materialization.predicted_future_loss_ns << '\n';
    require(warm.metrics.prefix_reuse_path == PrefixReusePath::PrivateEndpoint &&
                warm.metrics.prefix_cache_hit_tokens >= endpoint,
            "replayed reasoning closer rejected the warm generated endpoint");
    for (std::uint32_t slot = 0; slot < service.slot_states().size(); ++slot) {
        (void)service.slot_erase(slot);
    }

    // The old identity payload must not be loaded using the new digest semantics.
    std::string old_bytes = read_file(snapshot);
    require(old_bytes.size() > 12, "reasoning snapshot has no version field");
    const std::uint32_t old_version = 3;
    std::copy_n(reinterpret_cast<const char*>(&old_version), sizeof(old_version),
                old_bytes.begin() + 8);
    const auto old_snapshot = directory.path / "obsolete.nss";
    {
        std::ofstream output(old_snapshot, std::ios::binary);
        output.write(old_bytes.data(), old_bytes.size());
    }
    bool rejected = false;
    try {
        (void)service.slot_restore(0, old_snapshot.string());
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "obsolete private identity snapshot was accepted");

    (void)service.slot_restore(0, snapshot.string());
    const auto restored = run(history, 8);
    std::cout << "reasoning_continuation endpoint=" << endpoint
              << " warm=" << warm.metrics.prefix_cache_hit_tokens
              << " restored=" << restored.metrics.prefix_cache_hit_tokens
              << " paths=" << static_cast<unsigned>(warm.metrics.prefix_reuse_path) << '/'
              << static_cast<unsigned>(restored.metrics.prefix_reuse_path)
              << " drafted=" << warm.metrics.speculative_draft_tokens << '/'
              << restored.metrics.speculative_draft_tokens
              << " accepted=" << warm.metrics.speculative_accepted_tokens << '/'
              << restored.metrics.speculative_accepted_tokens
              << " warm_tokens=" << Json(warm.generated_token_ids)
              << " restored_tokens=" << Json(restored.generated_token_ids) << '\n';
    require(restored.metrics.prefix_reuse_path == PrefixReusePath::PrivateEndpoint &&
                restored.metrics.prefix_cache_hit_tokens >= endpoint &&
                restored.generated_token_ids == warm.generated_token_ids &&
                restored.metrics.speculative_draft_tokens ==
                    warm.metrics.speculative_draft_tokens &&
                restored.metrics.speculative_accepted_tokens ==
                    warm.metrics.speculative_accepted_tokens,
            "restored reasoning endpoint changed the continuation or MTP decisions");
}

void exercise_nonblocking_durable_ingress(const char* artifact, bool long_prefix = false) {
    std::cout << std::unitbuf;
    std::cout << "concurrent_ingress long_prefix=" << long_prefix << " phase=oracles\n";
    TemporaryDirectory temporary;
    ServeOptions configured    = options(artifact);
    configured.max_context     = long_prefix ? 32768 : 4096;
    configured.kv_capacity     = KvCapacityPolicy::explicit_capacity(long_prefix ? 98304 : 8192);
    configured.max_concurrency = 3;
    configured.max_pending_requests                            = 3;
    configured.context_cache.device_state_slots                = 3;
    configured.context_cache.host_state_slots                  = 8;
    configured.context_cache.max_private_continuations         = 6;
    configured.context_cache.max_shared_prefixes               = 3;
    configured.context_cache.max_long_anchors_per_continuation = 2;
    configured.shared_prefix_cache_dir                         = temporary.path / "shared";
    configured.shared_prefix_cache_workers                     = 1;
    configured.shared_prefix_cache_jobs                        = 2;
    configured.shared_prefix_cache_staging_bytes               = 2ULL << 30;

    const auto instructions = [long_prefix](std::string_view name) {
        std::string result(name);
        for (int index = 0; index < (long_prefix ? 19000 : 300); ++index) { result += " stable"; }
        return result + "\n=== CACHE_BREAKPOINT ===\nFollow the user's request.";
    };
    const auto request = [&](std::string_view name, bool long_output) {
        const Json body{
            {"model", "qwen3.8"},
            {"messages",
             Json::array(
                 {Json{{"role", "system"}, {"content", instructions(name)}},
                  Json{
                      {"role", "user"},
                      {"content",
                       long_output
                           ? "Write an extensive PHP class with at least 100 methods. Include all "
                             "method implementations. Continue writing code until the output limit."
                           : "Write the numbers from one through fifty, separated by spaces."}}})},
            {"max_tokens", long_output ? 1024 : 48}};
        return parse_chat_completion_request(body, RequestLimits{}).generation;
    };
    const auto wait_until = [](auto predicate, std::chrono::seconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    };

    std::array<GenerationOutcome, 2> oracles;
    {
        auto cold_options               = configured;
        cold_options.allow_prefix_reuse = false;
        GenerationService cold(cold_options);
        for (std::size_t index = 0; index < oracles.size(); ++index) {
            auto prepared  = cold.prepare(request(index == 0 ? "restore" : "cold", false),
                                          GenerationConsumerMode::Aggregate);
            oracles[index] = cold.run(prepared, nullptr);
        }
    }
    {
        std::cout << "concurrent_ingress phase=publisher\n";
        GenerationService publisher(configured);
        auto prepared =
            publisher.prepare(request("restore", false), GenerationConsumerMode::Aggregate);
        (void)publisher.run(prepared, nullptr);
        require(wait_until(
                    [&] {
                        const auto stats = publisher.runtime_stats();
                        return stats.shared_ssd_writes_completed != 0 &&
                               stats.shared_ssd_queued_jobs == 0 &&
                               stats.shared_ssd_active_jobs == 0 &&
                               stats.shared_ssd_pending_export_claims == 0;
                    },
                    std::chrono::seconds(30)),
                "ingress fixture did not publish its durable prefix");
    }

    std::cout << "concurrent_ingress phase=service\n";
    GenerationService service(configured);
    Engine& engine = testing::GenerationServiceTestAccess::engine(service);
    PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(ChatMessage{.role  = ChatRole::System,
                                         .parts = {MessagePart{.text = instructions("restore")}}});
    input.messages.push_back(
        ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = "Reply briefly."}}});
    auto prompt         = engine.prepare(std::move(input));
    using Access        = runtime::DurableSharedSnapshotAccess;
    const auto expected = Access::candidates(engine, prompt);
    require(!expected.empty(), "ingress fixture has no exact durable identity");
    std::future<std::vector<Access::Candidate>> discovery;
    std::future<MemorySummary> accounting;
    bool discovered_while_locked = false;
    bool observed_while_locked   = false;
    runtime::testing::SharedSnapshotTestAccess::with_execution_lock(engine, [&] {
        discovery =
            std::async(std::launch::async, [&] { return Access::candidates(engine, prompt); });
        accounting = std::async(std::launch::async, [&] { return service.memory_summary(); });
        discovered_while_locked =
            discovery.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        observed_while_locked =
            accounting.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    });
    // Release the execution boundary before consuming a possibly blocked future. A regression
    // must fail with a diagnosis, not deadlock teardown.
    const auto discovered = discovery.get();
    const auto memory     = accounting.get();
    require(discovered_while_locked,
            "durable identity discovery waited for the execution lock before FIFO submission");
    require(discovered == expected, "concurrent discovery changed exact durable identities");
    require(observed_while_locked,
            "request logging waited for the execution lock to observe memory accounting");
    require(memory.device_main_kv_capacity_pages != 0 && memory.host_state_capacity_slots == 8 &&
                memory.logical_state_capacity_slots != 0,
            "initial memory snapshot lost the configured physical/logical capacities");

    for (const std::string_view route : {"ssd", "warm", "cold"}) {
        std::cout << "concurrent_ingress route=" << route << " phase=start\n";
        const auto before = service.runtime_stats();
        std::atomic<bool> cancel_first{false};
        std::atomic<bool> first_output{false};
        std::atomic<bool> first_done{false};
        auto first = std::async(std::launch::async, [&] {
            struct Completion {
                std::atomic<bool>& flag;
                ~Completion() { flag.store(true); }
            } completion{first_done};
            auto prepared =
                service.prepare(request("active", true), GenerationConsumerMode::Streaming,
                                [&] { return cancel_first.load(); });
            StreamSink sink;
            sink.on_content   = [&](const std::string&) { first_output.store(true); };
            sink.is_cancelled = [&] { return cancel_first.load(); };
            return service.run(prepared, &sink);
        });

        struct CancelOnExit {
            std::atomic<bool>& flag;

            ~CancelOnExit() { flag.store(true); }
        } cleanup{cancel_first};

        require(wait_until(
                    [&] {
                        return first_done.load() ||
                               (first_output.load() &&
                                service.runtime_stats().decode_ready_requests != 0);
                    },
                    std::chrono::seconds(20)),
                "first ingress request never began decoding");
        require(!first_done.load(), "first ingress request finished before the overlap fixture");

        // A real control operation also needs an execution boundary. Repeated decode rounds
        // must hand it the boundary before finishing the whole generation; the same handoff
        // lets Gateway export completion release the pins needed by the incoming request.
        engine.reset_memory_peaks();
        require(!first_done.load(),
                "execution-boundary control was starved until the long generation completed");

        std::atomic<bool> observed_overlap{false};
        auto prepared = service.prepare(request(route == "cold" ? "cold" : "restore", false),
                                        GenerationConsumerMode::Streaming);
        // HttpServer reads this accounting when recording request_start before consuming the
        // stream. It must not wait until the unrelated long generation releases execution.
        const auto start_memory = service.memory_summary();
        require(start_memory.device_main_kv_occupied_pages != 0 &&
                    start_memory.device_main_kv_occupied_pages <=
                        start_memory.device_main_kv_capacity_pages,
                "request-start memory snapshot reports impossible KV occupancy");
        const auto prepared_at = service.runtime_stats();
        observed_overlap.store(prepared_at.running_requests >= 2);
        StreamSink sink;
        sink.on_content = [&](const std::string&) {
            if (service.runtime_stats().running_requests >= 2) { observed_overlap.store(true); }
        };
        const auto second          = service.run(prepared, &sink);
        const auto terminal_memory = service.memory_summary();
        require(terminal_memory.device_main_kv_occupied_pages <=
                    terminal_memory.device_main_kv_capacity_pages,
                "request-terminal memory snapshot reports impossible KV occupancy");
        const bool second_finished_first = !first_done.load();
        std::cout << "concurrent_ingress route=" << route << " phase=cancel-first\n";
        cancel_first.store(true);
        const auto first_result   = first.get();
        const auto after          = settled_stats(service, true);
        const auto settled_memory = service.memory_summary();
        require(settled_memory.logical_state_used_slots == after.logical_state_used_slots &&
                    settled_memory.device_main_kv_occupied_pages ==
                        after.device_main_kv_occupied_pages &&
                    settled_memory.host_state_occupied_slots == after.host_state_occupied_slots,
                "settled memory snapshot differs from published resource accounting");
        std::cout << "concurrent_ingress route=" << route
                  << " observed_overlap=" << observed_overlap.load()
                  << " second_finished_first=" << second_finished_first
                  << " prepare_seconds=" << second.metrics.prepare_seconds
                  << " queue_seconds=" << second.metrics.engine_timing.queue_wait_seconds
                  << " ttft_seconds=" << second.metrics.ttft_seconds << '\n';
        require(observed_overlap.load() && second_finished_first,
                "incoming service request did not progress before the long generation completed");
        require(after.decode_row_rounds - before.decode_row_rounds >
                    after.decode_rounds - before.decode_rounds,
                "overlapping service requests never formed a multi-row decode batch");
        require(first_result.metrics.speculative_rounds != 0,
                "overlap fixture did not execute real MTP decode");
        require(second.generated_token_ids == oracles[route == "cold" ? 1 : 0].generated_token_ids,
                "concurrent service continuation differs from cache-disabled oracle");
        if (route == "ssd") {
            require(second.metrics.durable_loaded_from_ssd &&
                        second.metrics.prefix_cache_hit_tokens != 0,
                    "concurrent SSD route did not restore its exact checkpoint");
        } else if (route == "warm") {
            require(second.metrics.durable_warm_available &&
                        !second.metrics.durable_loaded_from_ssd &&
                        second.metrics.prefix_cache_hit_tokens != 0,
                    "concurrent warm route lost the resident checkpoint");
        } else {
            require(second.metrics.prefix_reuse_path == PrefixReusePath::Root &&
                        second.metrics.prefix_cache_hit_tokens == 0,
                    "concurrent cold route reused a different harness");
        }
        require(after.logical_state_reserved_slots == 0 && after.logical_state_inflight_slots == 0,
                "concurrent ingress retained an unfinished resource reservation");
        std::cout << "concurrent_ingress route=" << route
                  << " overlap=1 batched_decode=1 oracle=exact"
                  << " prompt_tokens=" << second.prompt_tokens
                  << " reused_tokens=" << second.metrics.prefix_cache_hit_tokens
                  << " prepare_seconds=" << second.metrics.prepare_seconds
                  << " ttft_seconds=" << second.metrics.ttft_seconds << '\n';
    }
}

void exercise_changed_user_history(const char* artifact) {
    ServeOptions configured                                    = options(artifact);
    configured.context_cache.device_state_slots                = 3;
    configured.context_cache.max_long_anchors_per_continuation = 2;
    const std::string stable =
        "Keep replies concise.\n=== CACHE_BREAKPOINT ===\nHistory regression.";
    const std::string user      = "What is seven plus five?";
    const std::string annotated = user + "\n\n[CHANNEL: direct]\nReply directly in this response.";
    const Json initial          = Json::array({Json{{"role", "system"}, {"content", stable}},
                                               Json{{"role", "user"}, {"content", annotated}}});
    Json changed;
    GenerationOutcome rewritten;
    {
        GenerationService service(configured);
        const auto seed = generate_history(service, initial, true);
        Json appended   = initial;
        appended.push_back(Json{{"role", "assistant"}, {"content", seed.text}});
        appended.push_back(Json{{"role", "user"}, {"content", "And eight plus six?"}});
        const auto continued = generate_history(service, appended, true);
        require(continued.metrics.prefix_cache_hit_tokens >=
                    static_cast<std::uint32_t>(seed.prompt_tokens),
                "unchanged historical user message lost its conversation checkpoint");

        changed               = appended;
        changed[1]["content"] = user;
        rewritten             = generate_history(service, changed, true);
        require(rewritten.metrics.prefix_cache_hit_tokens <
                    static_cast<std::uint32_t>(seed.prompt_tokens),
                "rewritten historical user message incorrectly reused the old conversation head");
    }
    configured.allow_prefix_reuse = false;
    GenerationService oracle(configured);
    const auto cold = generate_history(oracle, changed, true);
    require(rewritten.generated_token_ids == cold.generated_token_ids,
            "changed-history fallback differs from the exact cold prompt");
}

void exercise_durable_two_lineage_warm_reuse(const char* artifact) {
    TemporaryDirectory temporary;
    ServeOptions configured                                    = options(artifact);
    configured.max_concurrency                                 = 3;
    configured.max_pending_requests                            = 3;
    configured.context_cache.device_state_slots                = 3;
    configured.context_cache.host_state_slots                  = 8;
    configured.context_cache.host_kv_capacity_bytes            = 2ULL << 30;
    configured.context_cache.max_private_continuations         = 6;
    configured.context_cache.max_shared_prefixes               = 3;
    configured.context_cache.max_long_anchors_per_continuation = 2;
    configured.shared_prefix_cache_dir                         = temporary.path / "shared-prefixes";
    configured.shared_prefix_cache_max_records                 = 8;
    configured.shared_prefix_cache_max_bytes                   = 8ULL << 30;
    configured.shared_prefix_cache_staging_bytes               = 2ULL << 30;
    configured.shared_prefix_cache_workers                     = 1;
    configured.shared_prefix_cache_jobs                        = 2;

    const auto instructions = [](std::string_view lineage) {
        std::string text = "durable-lineage-" + std::string(lineage) + ' ';
        for (std::uint32_t index = 0; index < 900; ++index) { text += "stable "; }
        text += "\n=== CACHE_BREAKPOINT ===\nvolatile workspace ";
        text += lineage;
        for (std::uint32_t index = 0; index < 100; ++index) { text += " suffix"; }
        return text;
    };
    const auto wait_for_writes = [](const GenerationService& service, std::uint64_t minimum) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            const RuntimeStats stats = service.runtime_stats();
            if (stats.shared_ssd_writes_completed >= minimum && stats.shared_ssd_queued_jobs == 0 &&
                stats.shared_ssd_active_jobs == 0 && stats.shared_ssd_pending_export_claims == 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };
    GenerationOutcome codex_oracle;
    {
        GenerationService seed(configured);
        (void)generate(seed, instructions("direct-a"), "Direct seed A.", false);
        codex_oracle = generate(seed, instructions("codex-b"), "Codex seed B.", true);
        require(wait_for_writes(seed, 2),
                "two-lineage fixture did not durably persist both SSD candidates");
    }

    GenerationService service(configured);
    const auto trace_state = [&](int request) {
        const RuntimeStats stats = settled_stats(service, true);
        std::cerr << "TRACE r" << request << " logical=" << stats.logical_state_used_slots << '+'
                  << stats.logical_state_reserved_slots << '/' << stats.logical_state_capacity_slots
                  << " device=" << stats.device_state_occupied_slots
                  << " host=" << stats.host_state_occupied_slots << '\n';
    };
    const GenerationOutcome direct_restored =
        generate(service, instructions("direct-a"), "Direct seed A.", false);
    trace_state(1);
    require(direct_restored.metrics.durable_loaded_from_ssd &&
                direct_restored.metrics.durable_fallback_reason == "ssd-successful-restore" &&
                direct_restored.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
                direct_restored.metrics.prefix_cache_hit_tokens != 0,
            "first Direct lineage did not restore its exact durable shared prefix");
    const GenerationOutcome codex_seed =
        generate(service, instructions("codex-b"), "Codex seed B.", true, false, "codex-lineage");
    trace_state(2);
    const Json codex_continuation = Json::array({
        Json{{"role", "system"}, {"content", instructions("codex-b")}},
        Json{{"role", "user"}, {"content", "Codex seed B."}},
        Json{{"role", "assistant"}, {"content", codex_seed.text}},
        Json{{"role", "user"}, {"content", "Continue Codex B."}},
    });
    (void)generate_history(service, codex_continuation, true, true, "codex-continuation");
    trace_state(3);
    (void)generate(service, instructions("direct-a"), "Direct follow-up A.", false, true);
    trace_state(4);
    const std::uint64_t writes_before_direct_c =
        service.runtime_stats().shared_ssd_writes_completed;
    const GenerationOutcome direct_c =
        generate(service, instructions("direct-c"), "Direct activity C.", false);
    trace_state(5);
    require(wait_for_writes(service, writes_before_direct_c + 1U),
            "Direct C shared prefix was not durably backed before further churn");
    const Json direct_c_continuation = Json::array({
        Json{{"role", "system"}, {"content", instructions("direct-c")}},
        Json{{"role", "user"}, {"content", "Direct activity C."}},
        Json{{"role", "assistant"}, {"content", direct_c.text}},
        Json{{"role", "user"}, {"content", "Continue Direct C."}},
    });
    (void)generate_history(service, direct_c_continuation, false, true);
    trace_state(6);
    (void)generate(service, instructions("direct-c"), "Direct activity C.", false, true);
    trace_state(7);
    require(wait_for_writes(service, 1),
            "Direct activity did not finish its durable exports before replacement");
    const RuntimeStats filled = settled_stats(service, true);
    // The next repeat must select a deep warm checkpoint with the complete State pool full.
    // Shared catalog occupancy is a policy choice; SSD replacement has a separate fixture.
    require(filled.logical_state_capacity_slots == 14 && filled.logical_state_used_slots == 14 &&
                filled.logical_state_reserved_slots == 0 &&
                filled.logical_state_inflight_slots == 0 &&
                filled.device_state_occupied_slots == 6 && filled.host_state_occupied_slots == 8,
            "Direct activity did not fill the Device/Host State pool before warm reuse");

    const std::uint64_t loads_before_direct_repeat =
        service.runtime_stats().shared_ssd_loads_completed;
    const GenerationOutcome direct_repeated =
        generate(service, instructions("direct-a"), "Direct seed A.", false, true);
    trace_state(8);
    std::cerr << "TRACE r8 reuse=" << direct_repeated.metrics.prefix_cache_hit_tokens
              << " path=" << static_cast<unsigned>(direct_repeated.metrics.prefix_reuse_path)
              << " durable=" << direct_repeated.metrics.durable_fallback_reason
              << " frontier=" << direct_repeated.metrics.durable_restore_frontier
              << " warm=" << direct_repeated.metrics.durable_warm_available << '\n';
    require(!direct_repeated.metrics.durable_loaded_from_ssd &&
                direct_repeated.metrics.durable_warm_available &&
                direct_repeated.metrics.durable_fallback_reason == "warm-source-selected" &&
                direct_repeated.metrics.prefix_cache_hit_tokens >
                    direct_restored.metrics.durable_restore_frontier &&
                direct_repeated.prompt_tokens >
                    static_cast<int>(direct_repeated.metrics.prefix_cache_hit_tokens) &&
                direct_repeated.prompt_tokens -
                        static_cast<int>(direct_repeated.metrics.prefix_cache_hit_tokens) ==
                    direct_repeated.prompt_tokens -
                        static_cast<int>(direct_repeated.metrics.durable_restore_frontier) &&
                service.runtime_stats().shared_ssd_loads_completed == loads_before_direct_repeat,
            "repeated Direct lineage did not choose its deepest warm checkpoint");
    // Consuming obsolete history during the Direct repeat can free a descriptor. Device State
    // remains full: require actual reclamation for Codex, plus exact output and deepest reuse,
    // instead of assuming the preceding request retains every obsolete descriptor.
    const RuntimeStats before_codex = settled_stats(service, true);
    const GenerationOutcome codex =
        generate(service, instructions("codex-b"), "Codex seed B.", true, false, "codex-lineage");
    trace_state(9);
    std::cerr << "TRACE r9 reuse=" << codex.metrics.prefix_cache_hit_tokens
              << " path=" << static_cast<unsigned>(codex.metrics.prefix_reuse_path)
              << " durable=" << codex.metrics.durable_fallback_reason
              << " frontier=" << codex.metrics.durable_restore_frontier
              << " warm=" << codex.metrics.durable_warm_available
              << " best=" << codex.metrics.materialization.best_reuse_prompt_tokens
              << " targets=" << codex.metrics.materialization.targets_evaluated << " stop="
              << materialization_stop_reason_name(codex.metrics.materialization.stop_reason)
              << " maximal=" << codex.metrics.materialization.selected_maximal_fallback
              << " reclaimed_state=" << codex.metrics.materialization.reclaimed_device_state_slots
              << '\n';
    require(
        !codex.metrics.durable_loaded_from_ssd && codex.metrics.durable_warm_available &&
            codex.metrics.durable_fallback_reason == "warm-source-selected" &&
            codex.metrics.prefix_reuse_path == PrefixReusePath::PrivateResponseReplay &&
            codex.metrics.prefix_cache_hit_tokens == codex.metrics.durable_restore_frontier &&
            codex.metrics.prefix_cache_hit_tokens > codex_seed.metrics.durable_restore_frontier &&
            codex.prompt_tokens > static_cast<int>(codex.metrics.prefix_cache_hit_tokens) &&
            codex.prompt_tokens - static_cast<int>(codex.metrics.prefix_cache_hit_tokens) ==
                codex.prompt_tokens - static_cast<int>(codex.metrics.durable_restore_frontier) &&
            codex.generated_token_ids == codex_oracle.generated_token_ids &&
            before_codex.logical_state_reserved_slots == 0 &&
            before_codex.logical_state_inflight_slots == 0 &&
            before_codex.device_state_occupied_slots == 6 &&
            codex.metrics.materialization.reclaimed_device_state_slots != 0,
        "Codex repeat did not choose the deepest warm checkpoint while reclaiming Device State");
    std::cout << "two_lineage_replay logical=" << before_codex.logical_state_used_slots << '/'
              << before_codex.logical_state_capacity_slots
              << " device=" << before_codex.device_state_occupied_slots
              << " host=" << before_codex.host_state_occupied_slots
              << " direct_reuse=" << direct_repeated.metrics.prefix_cache_hit_tokens
              << " codex_reuse=" << codex.metrics.prefix_cache_hit_tokens << " codex_suffix="
              << codex.prompt_tokens - static_cast<int>(codex.metrics.prefix_cache_hit_tokens)
              << '\n';
}

void exercise_ssd_replacement_beside_background(const char* artifact) {
    TemporaryDirectory temporary;
    auto configured                                    = options(artifact);
    configured.max_concurrency                         = 3;
    configured.max_pending_requests                    = 3;
    configured.kv_capacity                             = KvCapacityPolicy::explicit_capacity(8192);
    configured.context_cache.device_state_slots        = 6;
    configured.context_cache.max_private_continuations = 6;
    configured.context_cache.max_shared_prefixes       = 1;
    configured.shared_prefix_cache_dir                 = temporary.path / "ssd-side-request";
    // Exactly the side harness is durable. Background exports hit the record quota, so
    // its inactive shared entry has no SSD backing; the private head is independently live.
    configured.shared_prefix_cache_max_records = 1;
    configured.shared_prefix_cache_workers     = 1;
    const auto instructions                    = [](std::string name) {
        for (int i = 0; i < 400; ++i) { name += " stable"; }
        return name + "\n=== CACHE_BREAKPOINT ===\nAnswer the user.";
    };
    const auto incoming   = instructions("side-project");
    const auto resident   = instructions("background-project");
    const auto wait_until = [](auto predicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    };
    GenerationOutcome oracle;
    {
        GenerationService publisher(configured);
        oracle = generate(publisher, incoming, "Hi", false);
        require(wait_until([&] {
                    const auto stats = publisher.runtime_stats();
                    return stats.shared_ssd_writes_completed == 1 &&
                           stats.shared_ssd_active_jobs == 0 &&
                           stats.shared_ssd_pending_export_claims == 0;
                }),
                "side harness did not persist its sole SSD record");
    }
    GenerationService service(configured);
    const auto seed = generate(service, resident, "Reply briefly.", false, false, "background");
    require(wait_until([&] {
                const auto stats = service.runtime_stats();
                return stats.shared_ssd_quota_rejections != 0 &&
                       stats.shared_ssd_queued_jobs == 0 && stats.shared_ssd_active_jobs == 0 &&
                       stats.shared_ssd_pending_export_claims == 0;
            }),
            "background shared owner did not settle without SSD backing");
    Json history = Json::array({
        Json{{"role", "system"}, {"content", resident}},
        Json{{"role", "user"}, {"content", "Reply briefly."}},
        Json{{"role", "assistant"}, {"content", seed.text}},
        Json{{"role", "user"},
             {"content", "Write an extensive PHP class with at least 100 methods. Include all "
                         "implementations."}},
    });
    Json body{{"model", "qwen3.8"}, {"messages", history}, {"max_tokens", 1024}};
    auto background_request = parse_chat_completion_request(body, RequestLimits{}).generation;
    ContextCacheHints hints;
    hints.session_key = "background";
    std::atomic<bool> cancel{false}, started{false}, done{false};
    std::atomic<std::uint32_t> background_reuse{0};
    auto background = std::async(std::launch::async, [&] {
        struct Completion {
            std::atomic<bool>& flag;
            ~Completion() { flag.store(true); }
        } completion{done};
        auto prepared = service.prepare(
            background_request, GenerationConsumerMode::Streaming, [&] { return cancel.load(); },
            hints);
        StreamSink sink;
        sink.on_start = [&](const GenerationStart& start) {
            background_reuse.store(start.reused_prompt_tokens);
        };
        sink.on_content   = [&](const std::string&) { started.store(true); };
        sink.is_cancelled = [&] { return cancel.load(); };
        return service.run(prepared, &sink);
    });

    struct CancelOnExit {
        std::atomic<bool>& flag;

        ~CancelOnExit() { flag.store(true); }
    } cleanup{cancel};

    require(wait_until([&] { return started.load() || done.load(); }) && !done.load(),
            "background continuation did not remain active beside the side request");
    const auto expected_endpoint = seed.prompt_tokens + seed.generated_token_ids.size() - 1U;
    require(background_reuse.load() == expected_endpoint,
            "background fixture did not reuse its complete private endpoint");
    Json side_body{{"model", "qwen3.8"},
                   {"max_tokens", 8},
                   {"messages", Json::array({Json{{"role", "system"}, {"content", incoming}},
                                             Json{{"role", "user"}, {"content", "Hi"}}})}};
    auto side_request     = parse_chat_completion_request(side_body, RequestLimits{}).generation;
    auto prepared         = service.prepare(side_request, GenerationConsumerMode::Aggregate);
    const auto side       = service.run(prepared, nullptr);
    const bool overlapped = !done.load();
    cancel.store(true);
    (void)background.get();
    const auto stats            = settled_stats(service, true);
    const bool replaced_unsaved = std::any_of(
        side.checkpoint_lifecycle.begin(), side.checkpoint_lifecycle.end(), [](const auto& fact) {
            return fact.scope == CheckpointLifecycleScope::Shared &&
                   fact.operation == CheckpointLifecycleOperation::Evicted &&
                   fact.status == CheckpointLifecycleStatus::Committed &&
                   fact.destination_tier == CheckpointLifecycleTier::None;
        });
    require(
        overlapped && side.metrics.durable_loaded_from_ssd &&
            side.metrics.durable_fallback_reason == "ssd-successful-replacement" &&
            side.metrics.prefix_cache_hit_tokens != 0 && replaced_unsaved &&
            side.generated_token_ids == oracle.generated_token_ids &&
            side.metrics.speculative_draft_tokens == oracle.metrics.speculative_draft_tokens &&
            side.metrics.speculative_accepted_tokens ==
                oracle.metrics.speculative_accepted_tokens &&
            stats.logical_state_reserved_slots == 0 && stats.logical_state_inflight_slots == 0,
        "side request could not replace unused unsaved shared state beside a private continuation");
    const auto resumed = generate_history(service, history, false, false, "background");
    require(resumed.metrics.prefix_cache_hit_tokens == expected_endpoint &&
                resumed.metrics.prefix_reuse_path == PrefixReusePath::PrivateEndpoint,
            "SSD recovery disturbed the background session's successful endpoint");
    std::cout << "ssd_side_request background_reused=" << expected_endpoint
              << " side_reused=" << side.metrics.prefix_cache_hit_tokens
              << " active_background_preserved=true exact_oracle_match=true\n";
}

void exercise_ssd_replaces_memory_only_checkpoint(const char* artifact) {
    using Access = runtime::DurableSharedSnapshotAccess;
    TemporaryDirectory temporary;
    auto configured                              = options(artifact);
    configured.context_cache.device_state_slots  = 4;
    configured.context_cache.max_shared_prefixes = 1;
    configured.shared_prefix_cache_dir           = temporary.path / "ssd-memory-only-victim";
    const std::string incoming =
        "Stable harness " + std::string(1800, 'a') + "\n=== CACHE_BREAKPOINT ===\nAnswer the user.";
    GenerationOutcome oracle;
    std::shared_ptr<const std::vector<std::uint8_t>> incoming_bytes;
    std::string incoming_digest;
    {
        GenerationService publisher(configured);
        oracle              = generate(publisher, incoming, "Hi", false);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (publisher.runtime_stats().shared_ssd_writes_completed == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(publisher.runtime_stats().shared_ssd_writes_completed != 0,
                "memory-only replacement fixture did not persist its incoming harness");
        auto [slot, snapshot] = runtime::testing::SharedSnapshotTestAccess::export_first_durable(
            ninfer::serve::testing::GenerationServiceTestAccess::engine(publisher));
        (void)slot;
        snapshot.await_transfer(snapshot.bytes);
        incoming_bytes =
            std::make_shared<const std::vector<std::uint8_t>>(std::move(snapshot.bytes));
        incoming_digest = snapshot.content_digest;
    }
    GenerationService service(configured);
    std::string transient = "Transient user content";
    for (int i = 0; i < 400; ++i) { transient += " volatile"; }
    Json body{
        {"model", "qwen3.8"},
        {"max_tokens", 8},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content", Json::array({Json{{"type", "text"}, {"text", transient}},
                                      Json{{"type", "text"}, {"text", "Reply briefly."}}})}}})}};
    auto request = parse_chat_completion_request(body, RequestLimits{}).generation;
    ContextCacheHints hints;
    hints.allow_engine_automatic_shared_prefixes = false;
    hints.markers.push_back(PromptCacheMarker{
        .after_message_count      = 1,
        .kind                     = PromptCacheMarkerKind::SharedStablePrefix,
        .evidence                 = SharedCandidateEvidence::ExplicitBoundary,
        .location                 = PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 1,
    });
    auto prepared       = service.prepare(request, GenerationConsumerMode::Aggregate, {}, hints);
    const auto resident = service.run(prepared, nullptr);
    const bool created_shared =
        std::any_of(resident.checkpoint_lifecycle.begin(), resident.checkpoint_lifecycle.end(),
                    [](const auto& fact) {
                        return fact.scope == CheckpointLifecycleScope::Shared &&
                               fact.operation == CheckpointLifecycleOperation::Created &&
                               fact.status == CheckpointLifecycleStatus::Committed;
                    });
    require(created_shared && service.runtime_stats().shared_ssd_writes_completed == 0,
            "memory-only fixture did not create its non-durable shared victim");
    auto& engine = ninfer::serve::testing::GenerationServiceTestAccess::engine(service);
    require(resident.id_slot >= 0, "memory-only fixture did not retain its private endpoint");
    (void)settled_stats(service, true);
    engine.erase_slot(static_cast<std::uint32_t>(resident.id_slot));
    const auto baseline       = engine.runtime_stats();
    const auto same_ownership = [&](const RuntimeStats& observed) {
        return observed.context_cache_owners == baseline.context_cache_owners &&
               observed.device_state_occupied_slots == baseline.device_state_occupied_slots &&
               observed.host_state_occupied_slots == baseline.host_state_occupied_slots &&
               observed.device_main_kv_occupied_pages == baseline.device_main_kv_occupied_pages &&
               observed.device_backend_kv_occupied_pages ==
                   baseline.device_backend_kv_occupied_pages &&
               observed.host_kv_occupied_bytes == baseline.host_kv_occupied_bytes;
    };
    PromptInput incoming_input;
    incoming_input.messages = {
        ChatMessage{.role = ChatRole::System, .parts = {MessagePart{.text = incoming}}},
        ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = "Hi"}}},
    };
    incoming_input.options.enable_thinking   = false;
    incoming_input.options.preserve_thinking = true;
    const auto incoming_prompt               = engine.prepare(std::move(incoming_input));
    const auto candidates                    = Access::candidates(engine, incoming_prompt);
    const auto candidate =
        std::find_if(candidates.begin(), candidates.end(),
                     [&](const auto& item) { return item.content_digest == incoming_digest; });
    require(candidate != candidates.end(), "memory-only rollback lost its incoming SSD identity");
    RequestOptions request_options;
    request_options.execution.requested_output_tokens = 8;
    // Both seams run after imported allocations begin. The non-durable owner and its original
    // residency must be preserved or restored before failure escapes.
    for (const bool cancel : {false, true}) {
        const auto decision = Access::decide_recovery(engine, incoming_prompt, request_options,
                                                      std::span(&*candidate, 1));
        require(decision.source == Access::RecoverySource::Ssd && decision.replacement,
                "memory-only rollback fixture did not reserve a replacement");

        struct Gate {
            bool cancel;
            bool triggered = false;
            std::atomic<bool> cancelled{false};

            ~Gate() { runtime::testing::clear_shared_snapshot_import_gate(); }
        } gate{cancel};

        const runtime::testing::SharedSnapshotImportTestGate registration{
            .context = &gate,
            .checkpoint =
                [](void* context, runtime::testing::SharedSnapshotImportStage stage) {
                    auto& gate = *static_cast<Gate*>(context);
                    const auto selected =
                        gate.cancel ? runtime::testing::SharedSnapshotImportStage::StateAllocated
                                    : runtime::testing::SharedSnapshotImportStage::MainKvAllocated;
                    if (stage != selected) { return; }
                    gate.triggered = true;
                    if (gate.cancel) {
                        gate.cancelled.store(true);
                    } else {
                        throw std::bad_alloc();
                    }
                },
        };
        const auto pins_before = runtime::testing::shared_snapshot_export_pinned_sources();
        runtime::testing::install_shared_snapshot_import_gate(&registration);
        bool rejected = false;
        try {
            (void)Access::import(engine, decision.candidate, incoming_bytes,
                                 CancellationView([&] { return gate.cancelled.load(); }),
                                 decision.reservation_id);
        } catch (const std::bad_alloc&) { rejected = !cancel; } catch (const RequestError& error) {
            rejected = cancel && error.kind() == RequestErrorKind::Cancelled;
        }
        require(gate.triggered && rejected && engine.healthy() &&
                    same_ownership(engine.runtime_stats()) &&
                    runtime::testing::shared_snapshot_export_pinned_sources() == pins_before,
                "failed SSD import did not restore its memory-only victim and source pins");
    }
    auto restored_request = service.prepare(request, GenerationConsumerMode::Aggregate, {}, hints);
    const auto restored   = service.run(restored_request, nullptr);
    require(restored.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
                restored.metrics.prefix_cache_hit_tokens != 0 &&
                restored.generated_token_ids == resident.generated_token_ids &&
                restored.metrics.speculative_draft_tokens ==
                    resident.metrics.speculative_draft_tokens &&
                restored.metrics.speculative_accepted_tokens ==
                    resident.metrics.speculative_accepted_tokens,
            "rolled-back memory-only checkpoint lost exact generation or MTP state");
    const auto side = generate(service, incoming, "Hi", false);
    if (!side.metrics.durable_loaded_from_ssd) {
        std::cerr << "memory_only_victim fallback=" << side.metrics.durable_fallback_reason << '\n';
    }
    require(side.metrics.durable_loaded_from_ssd &&
                side.metrics.durable_fallback_reason == "ssd-successful-replacement" &&
                side.generated_token_ids == oracle.generated_token_ids &&
                side.metrics.speculative_draft_tokens == oracle.metrics.speculative_draft_tokens &&
                side.metrics.speculative_accepted_tokens ==
                    oracle.metrics.speculative_accepted_tokens,
            "in-memory rollback incorrectly required its victim to be SSD eligible");
    std::cout << "ssd_memory_only_victim reused=" << side.metrics.prefix_cache_hit_tokens
              << " allocation_rollback=true cancellation_rollback=true exact_oracle_match=true\n";
}

void exercise_service_ssd_winning_replacement(const char* artifact) {
    TemporaryDirectory temporary;
    ServeOptions configured                                    = options(artifact);
    configured.max_concurrency                                 = 3;
    configured.max_pending_requests                            = 3;
    configured.context_cache.device_state_slots                = 3;
    configured.context_cache.host_state_slots                  = 8;
    configured.context_cache.host_kv_capacity_bytes            = 2ULL << 30;
    configured.context_cache.max_private_continuations         = 6;
    configured.context_cache.max_shared_prefixes               = 3;
    configured.context_cache.max_long_anchors_per_continuation = 2;
    configured.auto_long_anchors                               = 1;
    configured.shared_prefix_cache_dir           = temporary.path / "service-ssd-replacement";
    configured.shared_prefix_cache_max_records   = 8;
    configured.shared_prefix_cache_max_bytes     = 8ULL << 30;
    configured.shared_prefix_cache_staging_bytes = 2ULL << 30;
    configured.shared_prefix_cache_workers       = 1;
    configured.shared_prefix_cache_jobs          = 2;

    const auto instructions = [](std::string_view lineage, std::uint32_t stable_words = 900) {
        std::string text = "service-ssd-lineage-" + std::string(lineage) + ' ';
        for (std::uint32_t index = 0; index < stable_words; ++index) { text += "stable "; }
        text += "\n=== CACHE_BREAKPOINT ===\nvolatile workspace ";
        text += lineage;
        for (std::uint32_t index = 0; index < 100; ++index) { text += " suffix"; }
        return text;
    };
    const auto wait_for_writes = [](const GenerationService& service, std::uint64_t minimum) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            const RuntimeStats stats = service.runtime_stats();
            if (stats.shared_ssd_writes_completed >= minimum && stats.shared_ssd_queued_jobs == 0 &&
                stats.shared_ssd_active_jobs == 0 && stats.shared_ssd_pending_export_claims == 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };
    const auto shared_owner_count = [](const GenerationService& service,
                                       const RuntimeStats& stats) {
        std::uint32_t all_owners = 0;
        for (std::uint8_t role = 0; role < static_cast<std::uint8_t>(ContextCacheMetricRole::Count);
             ++role) {
            for (std::uint8_t placement = 0;
                 placement < static_cast<std::uint8_t>(ContextCacheMetricPlacement::Count);
                 ++placement) {
                for (std::uint8_t pin = 0;
                     pin < static_cast<std::uint8_t>(ContextCacheMetricPin::Count); ++pin) {
                    for (std::uint8_t identity = 0;
                         identity < static_cast<std::uint8_t>(ContextCacheMetricIdentity::Count);
                         ++identity) {
                        all_owners += stats.context_cache_owners[context_cache_owner_metric_index(
                            static_cast<ContextCacheMetricRole>(role),
                            static_cast<ContextCacheMetricPlacement>(placement),
                            static_cast<ContextCacheMetricPin>(pin),
                            static_cast<ContextCacheMetricIdentity>(identity))];
                    }
                }
            }
        }
        const auto slots          = service.slot_states();
        const auto private_owners = static_cast<std::uint32_t>(std::count_if(
            slots.begin(), slots.end(), [](const auto& slot) { return slot.retained; }));
        require(all_owners >= private_owners, "service replay lost a private continuation owner");
        return all_owners - private_owners;
    };

    const std::string incoming_instructions   = instructions("incoming");
    constexpr std::uint32_t expected_frontier = 917;
    constexpr int expected_suffix_tokens      = 123;
    GenerationOutcome oracle;
    {
        GenerationService publisher(configured);
        oracle = generate(publisher, incoming_instructions, "SSD winner request.", false);
        require(wait_for_writes(publisher, 1),
                "service replay did not publish its incoming SSD checkpoint");
    }

    GenerationService service(configured);
    const GenerationOutcome initial =
        generate(service, incoming_instructions, "SSD winner request.", false);
    const auto initial_restore =
        std::find_if(initial.checkpoint_lifecycle.begin(), initial.checkpoint_lifecycle.end(),
                     [](const CheckpointLifecycleFact& fact) {
                         return fact.operation == CheckpointLifecycleOperation::Restored &&
                                fact.scope == CheckpointLifecycleScope::Shared &&
                                fact.source_tier == CheckpointLifecycleTier::Ssd &&
                                fact.destination_tier == CheckpointLifecycleTier::Host &&
                                fact.status == CheckpointLifecycleStatus::Committed;
                     });
    require(initial.metrics.durable_loaded_from_ssd &&
                initial.metrics.durable_fallback_reason == "ssd-successful-restore" &&
                initial.metrics.durable_restore_frontier == expected_frontier &&
                initial_restore != initial.checkpoint_lifecycle.end() &&
                initial_restore->frontier == expected_frontier &&
                !initial_restore->content_digest.empty() &&
                initial.generated_token_ids == oracle.generated_token_ids,
            "service replay did not begin with the original SSD recovery");
    require(initial.id_slot >= 0 && !initial.session_digest.empty() &&
                service.slot_erase(static_cast<std::uint32_t>(initial.id_slot),
                                   initial.session_digest) != 0,
            "service replay did not retire the first request's warm continuation");
    const GenerationOutcome resident_a =
        generate(service, instructions("resident-a", 1400), "Resident A.", false);
    const GenerationOutcome resident_b =
        generate(service, instructions("resident-b", 1400), "Resident B.", false);
    require(resident_a.id_slot >= 0 && !resident_a.session_digest.empty(),
            "service replay did not retain resident activity");
    const auto generate_explicit_resident = [&](std::string word, std::string suffix,
                                                std::string session_key) {
        std::string content;
        for (std::uint32_t index = 0; index < 1400; ++index) { content += word + ' '; }
        Json body{{"model", "qwen3.8"},
                  {"messages",
                   Json::array({Json{
                       {"role", "user"},
                       {"content", Json::array({Json{{"type", "text"}, {"text", content}},
                                                Json{{"type", "text"}, {"text", suffix}}})}}})},
                  {"max_tokens", 8}};
        GenerationRequest generation =
            parse_chat_completion_request(body, RequestLimits{}).generation;
        ContextCacheHints hints;
        hints.session_key                            = std::move(session_key);
        hints.allow_engine_automatic_shared_prefixes = false;
        hints.markers.push_back(PromptCacheMarker{
            .after_message_count      = 1,
            .kind                     = PromptCacheMarkerKind::SharedStablePrefix,
            .evidence                 = SharedCandidateEvidence::ExplicitBoundary,
            .location                 = PromptCacheMarkerLocation::MessagePartBoundary,
            .after_message_part_count = 1,
        });
        auto prepared =
            service.prepare(generation, GenerationConsumerMode::Aggregate, {}, std::move(hints));
        return service.run(prepared, nullptr);
    };
    const GenerationOutcome explicit_c =
        generate_explicit_resident("resident", "explicit resident C", "churn-lineage");
    require(explicit_c.id_slot >= 0,
            "service replay did not retain its shared-displacement activity");
    require(wait_for_writes(service, 2),
            "service replay did not durably settle its resident owners");
    const RuntimeStats before         = settled_stats(service, true);
    const MemorySummary before_memory = service.memory_summary();
    require(shared_owner_count(service, before) == 3,
            "service replay did not fill the three-cell shared catalog");

    const std::uint64_t loads_before = before.shared_ssd_loads_completed;
    const GenerationOutcome restored =
        generate(service, incoming_instructions, "SSD winner request.", false);
    const RuntimeStats after = settled_stats(service, true);
    const auto published =
        std::find_if(restored.checkpoint_lifecycle.begin(), restored.checkpoint_lifecycle.end(),
                     [&](const CheckpointLifecycleFact& fact) {
                         return fact.operation == CheckpointLifecycleOperation::Restored &&
                                fact.scope == CheckpointLifecycleScope::Shared &&
                                fact.source_tier == CheckpointLifecycleTier::Ssd &&
                                fact.destination_tier == CheckpointLifecycleTier::Host &&
                                fact.status == CheckpointLifecycleStatus::Committed &&
                                fact.content_digest == initial_restore->content_digest;
                     });
    const auto displaced = published != restored.checkpoint_lifecycle.begin() &&
                                   published != restored.checkpoint_lifecycle.end()
                               ? std::prev(published)
                               : restored.checkpoint_lifecycle.end();
    const auto identifies_preexisting_shared_owner = [&](const CheckpointLifecycleFact& victim) {
        const std::array prior_outcomes{&initial, &resident_a, &resident_b, &explicit_c};
        return std::any_of(prior_outcomes.begin(), prior_outcomes.end(), [&](const auto* outcome) {
            return std::any_of(outcome->checkpoint_lifecycle.begin(),
                               outcome->checkpoint_lifecycle.end(), [&](const auto& fact) {
                                   return fact.scope == CheckpointLifecycleScope::Shared &&
                                          fact.status == CheckpointLifecycleStatus::Committed &&
                                          fact.key_digests == victim.key_digests &&
                                          fact.frontier == victim.frontier;
                               });
        });
    };
    const int suffix_tokens =
        restored.prompt_tokens - static_cast<int>(restored.metrics.prefix_cache_hit_tokens);
    require(
        before.logical_state_capacity_slots == 14 && before.logical_state_used_slots == 11 &&
            before.logical_state_reserved_slots == 0 && before.logical_state_inflight_slots == 0 &&
            before.device_state_occupied_slots == 6 && before.host_state_occupied_slots == 5 &&
            before.shared_active_references == 0 && before_memory.host_state_capacity_slots == 8 &&
            loads_before == 1 && restored.metrics.durable_loaded_from_ssd &&
            !restored.metrics.durable_warm_available &&
            restored.metrics.durable_fallback_reason == "ssd-successful-replacement" &&
            restored.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
            restored.metrics.prefix_cache_hit_tokens == restored.metrics.durable_restore_frontier &&
            restored.metrics.durable_restore_frontier == expected_frontier &&
            suffix_tokens == expected_suffix_tokens &&
            restored.generated_token_ids == oracle.generated_token_ids &&
            after.shared_ssd_loads_completed == loads_before + 1U &&
            after.logical_state_capacity_slots == before.logical_state_capacity_slots &&
            after.logical_state_used_slots == 14 && after.logical_state_reserved_slots == 0 &&
            after.logical_state_inflight_slots == 0 &&
            after.device_state_occupied_slots == before.device_state_occupied_slots &&
            after.host_state_occupied_slots == before_memory.host_state_capacity_slots &&
            after.shared_active_references == 0 && shared_owner_count(service, after) == 3 &&
            displaced != restored.checkpoint_lifecycle.end() &&
            published != restored.checkpoint_lifecycle.end() &&
            displaced->operation == CheckpointLifecycleOperation::Evicted &&
            displaced->scope == CheckpointLifecycleScope::Shared &&
            displaced->source_tier == CheckpointLifecycleTier::Host &&
            displaced->destination_tier == CheckpointLifecycleTier::Ssd &&
            displaced->status == CheckpointLifecycleStatus::Committed &&
            displaced->frontier > expected_frontier &&
            displaced->key_digests != std::array<std::uint64_t, 2>{} &&
            displaced->key_digests != published->key_digests &&
            published->frontier == expected_frontier &&
            published->content_digest == initial_restore->content_digest &&
            identifies_preexisting_shared_owner(*displaced),
        "service-level 6/6 Device, 5/8 Host replay did not commit SSD-winning replacement");
    require(std::string_view(ninfer::build::revision) != "unknown" &&
                std::string_view(ninfer::build::revision).size() == 40,
            "service replay binary does not record an exact source revision");
    std::cout << "service_ssd_replacement build_revision=" << ninfer::build::revision
              << " source_dirty=" << ninfer::build::source_dirty
              << " frontier=" << restored.metrics.durable_restore_frontier
              << " suffix=" << suffix_tokens << " logical=" << before.logical_state_used_slots
              << '/' << before.logical_state_capacity_slots
              << " device=" << before.device_state_occupied_slots
              << " host=" << before.host_state_occupied_slots << '/'
              << before_memory.host_state_capacity_slots << '\n';
}

void exercise_deferred_durable_deadline_settlement(const char* artifact) {
    TemporaryDirectory temporary;
    ServeOptions configured                      = options(artifact);
    configured.max_concurrency                   = 1;
    configured.max_pending_requests              = 2;
    configured.context_cache.device_state_slots  = 2;
    configured.context_cache.host_state_slots    = 2;
    configured.context_cache.max_shared_prefixes = 1;
    configured.shared_prefix_cache_dir           = temporary.path / "deadline-prefixes";
    configured.shared_prefix_cache_max_records   = 2;
    configured.shared_prefix_cache_max_bytes     = 4ULL << 30;
    configured.shared_prefix_cache_staging_bytes = 2ULL << 30;
    configured.shared_prefix_cache_workers       = 1;
    configured.shared_prefix_cache_jobs          = 1;
    std::string stable                           = "deferred-deadline ";
    for (std::uint32_t index = 0; index < 900; ++index) { stable += "stable "; }
    stable += "\n=== CACHE_BREAKPOINT ===\nvolatile deadline suffix";
    const Json target_body{
        {"model", "qwen3.8"},
        {"messages", Json::array({Json{{"role", "system"}, {"content", stable}},
                                  Json{{"role", "user"}, {"content", "Reply yes."}}})},
        {"max_tokens", 3}};
    const auto target_request = [&] {
        return parse_chat_completion_request(target_body, RequestLimits{}).generation;
    };
    {
        GenerationService seed(configured);
        PreparedRequest prepared =
            seed.prepare(target_request(), GenerationConsumerMode::Aggregate);
        (void)seed.run(prepared, nullptr);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            const RuntimeStats stats = seed.runtime_stats();
            if (stats.shared_ssd_writes_completed != 0 && stats.shared_ssd_queued_jobs == 0 &&
                stats.shared_ssd_active_jobs == 0 && stats.shared_ssd_pending_export_claims == 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        require(seed.runtime_stats().shared_ssd_writes_completed != 0,
                "deferred deadline fixture did not persist its SSD candidate");
    }

    const auto verify_terminal = [](const ApiException& exception, std::string_view code) {
        return exception.error().code == code &&
               std::any_of(exception.error().checkpoint_lifecycle.begin(),
                           exception.error().checkpoint_lifecycle.end(), [](const auto& fact) {
                               return fact.operation == CheckpointLifecycleOperation::Restored &&
                                      fact.source_tier == CheckpointLifecycleTier::Ssd &&
                                      fact.status == CheckpointLifecycleStatus::Aborted;
                           });
    };
    const auto verify_released = [](const GenerationService& service, std::string_view label) {
        const RuntimeStats settled = settled_stats(service, true);
        if (settled.logical_state_reserved_slots != 0 ||
            settled.logical_state_inflight_slots != 0 || settled.waiting_requests != 0) {
            throw std::runtime_error(std::string(label) +
                                     " retained deferred recovery reservations");
        }
    };

    configured.pending_timeout_ms = 1000;
    for (const GenerationConsumerMode mode :
         {GenerationConsumerMode::Aggregate, GenerationConsumerMode::Streaming}) {
        GenerationService service(configured);
        testing::GenerationServiceTestAccess::set_before_payload_read(
            service, [] { std::this_thread::sleep_for(std::chrono::milliseconds(1500)); });
        bool deadline_reported = false;
        try {
            (void)service.prepare(target_request(), mode);
        } catch (const ApiException& exception) {
            deadline_reported = verify_terminal(exception, "request_queue_timeout");
        }
        require(deadline_reported, "deferred SSD deadline lost its mode-correct service lifecycle");
        verify_released(service, "deferred SSD deadline");
    }

    configured.pending_timeout_ms = 30000;
    GenerationService cancelled_service(configured);
    std::atomic<bool> cancelled{false};
    testing::GenerationServiceTestAccess::set_before_payload_read(
        cancelled_service, [&] { cancelled.store(true, std::memory_order_release); });
    bool cancellation_reported = false;
    try {
        (void)cancelled_service.prepare(target_request(), GenerationConsumerMode::Aggregate,
                                        [&] { return cancelled.load(std::memory_order_acquire); });
    } catch (const ApiException& exception) {
        cancellation_reported = verify_terminal(exception, "client_disconnected");
    }
    require(cancellation_reported,
            "aggregate deferred SSD cancellation lost its service lifecycle");
    verify_released(cancelled_service, "aggregate deferred SSD cancellation");
}

std::string base64(std::span<const std::uint8_t> bytes) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size() + 2U) / 3U * 4U);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3U) {
        const std::uint32_t first  = bytes[offset];
        const std::uint32_t second = offset + 1U < bytes.size() ? bytes[offset + 1U] : 0U;
        const std::uint32_t third  = offset + 2U < bytes.size() ? bytes[offset + 2U] : 0U;
        const std::uint32_t value  = (first << 16U) | (second << 8U) | third;
        encoded.push_back(alphabet[(value >> 18U) & 63U]);
        encoded.push_back(alphabet[(value >> 12U) & 63U]);
        encoded.push_back(offset + 1U < bytes.size() ? alphabet[(value >> 6U) & 63U] : '=');
        encoded.push_back(offset + 2U < bytes.size() ? alphabet[value & 63U] : '=');
    }
    return encoded;
}

std::vector<std::uint8_t> capture_ppm() {
    constexpr int width      = 32;
    constexpr int height     = 32;
    const std::string header = "P6\n32 32\n255\n";
    std::vector<std::uint8_t> image(header.begin(), header.end());
    image.reserve(image.size() + width * height * 3U);
    for (int pixel = 0; pixel < width * height; ++pixel) {
        image.push_back(static_cast<std::uint8_t>(pixel));
        image.push_back(static_cast<std::uint8_t>(pixel * 3));
        image.push_back(static_cast<std::uint8_t>(pixel * 7));
    }
    return image;
}

struct CapturedGeneration {
    GenerationOutcome outcome;
    std::string prompt;
    std::string response;
};

CapturedGeneration capture_generation(GenerationService& service, JsonlRequestLog& writer,
                                      GenerationRequest request, std::string protocol,
                                      std::uint64_t request_id, bool stream) {
    PreparedRequest prepared  = service.prepare(request, stream ? GenerationConsumerMode::Streaming
                                                                : GenerationConsumerMode::Aggregate);
    RequestLogContext context = make_request_log_context(
        request_id, std::move(protocol), request,
        RequestLogMetadata{.model       = "qwen3.8",
                           .response_id = "capture-response-" + std::to_string(request_id),
                           .stream      = stream,
                           .output_tokens_explicit = true},
        prepared);
    writer.write_request_start(context);
    StreamSink sink;
    GenerationOutcome outcome = service.run(prepared, stream ? &sink : nullptr);
    writer.write_request_done(context, outcome);
    const std::string suffix =
        writer.server_instance_id() + "-request-" + std::to_string(request_id) + ".md";
    return {.outcome = std::move(outcome),
            .prompt  = read_file(service.options().request_log_content_dir / ("prompt" + suffix)),
            .response =
                read_file(service.options().request_log_content_dir / ("response" + suffix))};
}

void exercise_protocol_content_capture(const char* artifact) {
    TemporaryDirectory temporary;
    ServeOptions configured            = options(artifact);
    configured.request_log_jsonl       = temporary.path / "requests.jsonl";
    configured.request_log_content_dir = temporary.path / "content";
    configured.context_cache           = ContextCacheOptions{.enabled = false};
    configured.enable_vision           = true;
    GenerationService service(configured);
    JsonlRequestLog writer(configured.request_log_jsonl, artifact, {},
                           configured.request_log_content_dir);

    const std::vector<std::uint8_t> media_bytes = capture_ppm();
    const std::string media_payload             = base64(media_bytes);
    const std::string data_uri = "data:image/x-portable-pixmap;base64," + media_payload;
    const Json chat_body{
        {"model", "qwen3.8"},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content", Json::array({Json{{"type", "text"}, {"text", "capture-chat-sentinel"}},
                                      Json{{"type", "image_url"},
                                           {"image_url", Json{{"url", data_uri}}}}})}}})},
        {"max_tokens", 4}};
    GenerationRequest chat = parse_chat_completion_request(chat_body, RequestLimits{}).generation;
    const CapturedGeneration chat_aggregate =
        capture_generation(service, writer, chat, "openai_chat_completions", 1, false);
    GenerationRequest chat_stream =
        parse_chat_completion_request(chat_body, RequestLimits{}).generation;
    const CapturedGeneration streamed = capture_generation(service, writer, std::move(chat_stream),
                                                           "openai_chat_completions", 2, true);

    OpenAIResponsesStore store(4, 1ULL << 20);
    const auto responses_request = parse_openai_responses_create_request(
        Json{{"model", "qwen3.8"},
             {"input",
              Json::array({
                  Json{{"role", "user"},
                       {"content",
                        Json::array(
                            {Json{{"type", "input_text"}, {"text", "capture-responses-sentinel"}},
                             Json{{"type", "input_image"}, {"image_url", data_uri}}})}},
              })},
             {"store", false},
             {"max_output_tokens", 4}},
        RequestLimits{});
    auto responses = resolve_openai_responses_prompt(responses_request.prompt, store,
                                                     "capture-responses", false);
    const CapturedGeneration responses_capture = capture_generation(
        service, writer, std::move(responses.generation), "openai_responses", 3, false);

    AnthropicThinkingSigner::Key signing_key{};
    AnthropicThinkingSigner signer(signing_key);
    GenerationRequest anthropic =
        parse_anthropic_messages_request(
            Json{{"model", "qwen3.8"},
                 {"messages",
                  Json::array({Json{
                      {"role", "user"},
                      {"content",
                       Json::array({Json{{"type", "text"}, {"text", "capture-anthropic-sentinel"}},
                                    Json{{"type", "image"},
                                         {"source", Json{{"type", "base64"},
                                                         {"media_type", "image/x-portable-pixmap"},
                                                         {"data", media_payload}}}}})}}})},
                 {"max_tokens", 4}},
            RequestLimits{}, signer)
            .generation;
    const CapturedGeneration anthropic_capture =
        capture_generation(service, writer, std::move(anthropic), "anthropic_messages", 4, false);

    require(chat_aggregate.prompt.find("capture-chat-sentinel") != std::string::npos &&
                responses_capture.prompt.find("capture-responses-sentinel") != std::string::npos &&
                anthropic_capture.prompt.find("capture-anthropic-sentinel") != std::string::npos,
            "protocol capture did not persist Frontend-rendered prompt semantics");
    require(chat_aggregate.prompt.find("\"messages\"") == std::string::npos &&
                responses_capture.prompt.find("\"input\"") == std::string::npos,
            "protocol capture persisted raw HTTP JSON instead of rendered prompt text");
    require(chat_aggregate.prompt.find("## Non-text media") != std::string::npos &&
                responses_capture.prompt.find("## Non-text media") != std::string::npos &&
                anthropic_capture.prompt.find("## Non-text media") != std::string::npos &&
                chat_aggregate.prompt.find("media_type=`image/x-portable-pixmap`") !=
                    std::string::npos &&
                responses_capture.prompt.find("media_type=`image/x-portable-pixmap`") !=
                    std::string::npos &&
                anthropic_capture.prompt.find("media_type=`image/x-portable-pixmap`") !=
                    std::string::npos &&
                chat_aggregate.prompt.find(media_payload) == std::string::npos &&
                responses_capture.prompt.find(data_uri) == std::string::npos &&
                anthropic_capture.prompt.find(media_payload) == std::string::npos,
            "protocol-acquired media was not reduced to safe type/size/digest metadata");
    require(streamed.outcome.text == chat_aggregate.outcome.text &&
                streamed.outcome.reasoning == chat_aggregate.outcome.reasoning &&
                streamed.response == chat_aggregate.response,
            "streaming and aggregate generation produced different final Markdown");
    require(chat_aggregate.response == format_response_markdown(chat_aggregate.outcome) &&
                responses_capture.response == format_response_markdown(responses_capture.outcome) &&
                anthropic_capture.response == format_response_markdown(anthropic_capture.outcome),
            "a protocol capture did not preserve the final logical response");
    const std::string jsonl = read_file(configured.request_log_jsonl);
    require(jsonl.find(media_payload) == std::string::npos &&
                jsonl.find("capture-chat-sentinel") == std::string::npos &&
                jsonl.find("capture-responses-sentinel") == std::string::npos &&
                jsonl.find("capture-anthropic-sentinel") == std::string::npos,
            "JSONL embedded acquired media or model-visible text");
}

int reserve_loopback_port() {
    const int socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) { throw std::runtime_error("failed to create port reservation socket"); }
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(socket_fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(socket_fd);
        throw std::runtime_error("failed to reserve loopback port");
    }
    socklen_t length = sizeof(address);
    if (::getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        ::close(socket_fd);
        throw std::runtime_error("failed to inspect loopback port");
    }
    const int port = ntohs(address.sin_port);
    ::close(socket_fd);
    return port;
}

void exercise_http_engine_failure(const char* artifact) {
    using Access = runtime::testing::SharedSnapshotTestAccess;
    std::vector<std::uint8_t> bytes;
    {
        auto configured = options(artifact);
        configured.context_cache.device_state_slots = 4;
        GenerationService publisher(configured);
        (void)generate(publisher,
                       "Stable harness " + std::string(1800, 'a') +
                           "\n=== CACHE_BREAKPOINT ===\nAnswer briefly.",
                       "Hi", false);
        (void)settled_stats(publisher, true);
        auto [slot, snapshot] = Access::export_first_durable(
            testing::GenerationServiceTestAccess::engine(publisher));
        (void)slot;
        snapshot.await_transfer(snapshot.bytes);
        bytes = std::move(snapshot.bytes);
        snapshot.release_storage();
    }
    for (const std::uint32_t stats_interval : {0U, 60000U}) {
        auto configured = options(artifact);
        configured.host = "127.0.0.1";
        configured.port = reserve_loopback_port();
        configured.log_stats_interval_ms = stats_interval;
        configured.context_cache.device_state_slots = 4;
        std::ostringstream logs;
        auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(logs);
        auto logger = std::make_shared<spdlog::logger>("engine-failure-test", sink);
        GenerationService service(configured, {}, logger);
        auto& engine = testing::GenerationServiceTestAccess::engine(service);

        HttpServer server(configured, logger);
        require(server.bind(), "failed to bind engine-failure regression server");
        server.attach(service);
        std::promise<bool> stopped;
        auto result = stopped.get_future();
        struct Listener {
            HttpServer& server;
            std::thread thread;
            ~Listener() {
                server.stop();
                if (thread.joinable()) { thread.join(); }
            }
        } listener{server, std::thread([&] {
                       try { stopped.set_value(server.listen()); }
                       catch (...) { stopped.set_exception(std::current_exception()); }
                   })};
        httplib::Client client(configured.host, configured.port);
        client.set_read_timeout(5);
        const auto healthy = client.Get("/health");
        require(healthy && healthy->status == 200, "healthy engine did not serve HTTP");
        const auto invalid = client.Post("/v1/chat/completions", "{}", "application/json");
        require(invalid && invalid->status == 400 && service.healthy() &&
                    result.wait_for(std::chrono::milliseconds(500)) == std::future_status::timeout,
                "recoverable request rejection stopped the server");

        struct ImportFailure {
            runtime::testing::SharedSnapshotImportTestGate gate{
                .checkpoint = [](void*, runtime::testing::SharedSnapshotImportStage stage) {
                    if (stage == runtime::testing::SharedSnapshotImportStage::StateAllocated) {
                        throw std::logic_error("injected fatal engine failure");
                    }
                }};
            ImportFailure() { runtime::testing::install_shared_snapshot_import_gate(&gate); }
            ~ImportFailure() { runtime::testing::clear_shared_snapshot_import_gate(); }
        } failure;
        bool injected = false;
        try { (void)Access::import(engine, bytes); }
        catch (const std::logic_error&) { injected = true; }
        require(injected && !service.healthy(), "fixture did not latch an engine failure");
        // No health probe or new inference request is needed to shut the listener down.
        require(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "failed engine left the HTTP listener alive");
        require(!result.get(), "fatal engine shutdown was reported as successful");
        require(!client.Get("/health"), "failed engine continued accepting HTTP connections");
        std::cout << "engine_failure_shutdown stats_interval=" << stats_interval << " ok\n";
    }
}

void exercise_http_secret_exclusion(const char* artifact) {
    TemporaryDirectory temporary;
    constexpr std::string_view authorization_secret = "authorization-secret-6cb22c";
    constexpr std::string_view cookie_secret        = "cookie-secret-11ca3a";
    constexpr std::string_view signed_query_secret  = "signed-query-secret-bdc997";

    ServeOptions configured            = options(artifact);
    configured.host                    = "127.0.0.1";
    configured.port                    = reserve_loopback_port();
    configured.api_key                 = authorization_secret;
    configured.request_log_jsonl       = temporary.path / "requests.jsonl";
    configured.request_log_content_dir = temporary.path / "content";
    configured.context_cache           = ContextCacheOptions{.enabled = false};
    configured.enable_vision           = true;
    configured.log_stats_interval_ms   = 0;

    std::ostringstream operational;
    auto sink   = std::make_shared<spdlog::sinks::ostream_sink_mt>(operational);
    auto logger = std::make_shared<spdlog::logger>("capture-secret-test", sink);
    logger->set_pattern("%v");
    GenerationService service(configured, {}, logger);
    HttpServer server(configured, logger);
    require(server.bind(), "failed to bind HTTP privacy regression server");
    server.attach(service);

    struct Listener {
        HttpServer* server = nullptr;
        std::thread thread;

        ~Listener() {
            server->stop();
            if (thread.joinable()) { thread.join(); }
        }
    } listener{.server = &server, .thread = std::thread([&] { (void)server.listen(); })};

    const std::string signed_url =
        "http://127.0.0.1:" + std::to_string(configured.port) +
        "/private.ppm?X-Amz-Signature=" + std::string(signed_query_secret);
    const Json body{
        {"model", server.public_model_id()},
        {"messages",
         Json::array(
             {Json{{"role", "user"},
                   {"content", Json::array({Json{{"type", "text"}, {"text", "secret-url-request"}},
                                            Json{{"type", "image_url"},
                                                 {"image_url", Json{{"url", signed_url}}}}})}}})},
        {"max_tokens", 1}};
    httplib::Client client(configured.host, configured.port);
    httplib::Headers headers{{"Authorization", "Bearer " + std::string(authorization_secret)},
                             {"Cookie", "session=" + std::string(cookie_secret)}};
    const std::string media_payload = base64(capture_ppm());
    const std::string data_uri      = "data:image/x-portable-pixmap;base64," + media_payload;
    const Json accepted_body{
        {"model", server.public_model_id()},
        {"messages",
         Json::array(
             {Json{{"role", "user"},
                   {"content", Json::array({Json{{"type", "text"}, {"text", "http-safe-media"}},
                                            Json{{"type", "image_url"},
                                                 {"image_url", Json{{"url", data_uri}}}}})}}})},
        {"max_tokens", 1}};
    const httplib::Result accepted =
        client.Post("/v1/chat/completions", headers, accepted_body.dump(), "application/json");
    require(accepted && accepted->status == 200,
            "credential-bearing HTTP media request did not publish capture files");

    const httplib::Result response =
        client.Post("/v1/chat/completions", headers, body.dump(), "application/json");
    require(response && response->status == 400,
            "signed media URL did not reach deterministic acquisition rejection");
    const Json response_body = Json::parse(response->body);
    require(response_body.at("error").at("code") == "invalid_media",
            "signed media URL was not rejected by the media acquisition boundary");

    logger->flush();
    std::string markdown;
    std::error_code directory_error;
    if (std::filesystem::exists(configured.request_log_content_dir, directory_error)) {
        for (const auto& entry :
             std::filesystem::directory_iterator(configured.request_log_content_dir)) {
            if (entry.is_regular_file()) { markdown += read_file(entry.path()); }
        }
    }
    const std::string jsonl = read_file(configured.request_log_jsonl);
    const std::string logs  = operational.str();
    const std::string wire  = response->body;
    require(markdown.find("## Non-text media") != std::string::npos &&
                markdown.find("kind=`image`") != std::string::npos &&
                markdown.find("media_type=`image/x-portable-pixmap`") != std::string::npos &&
                markdown.find("bytes=`3085`") != std::string::npos &&
                markdown.find("sha256=`") != std::string::npos &&
                markdown.find(media_payload) == std::string::npos &&
                jsonl.find(media_payload) == std::string::npos,
            "HTTP media capture lost safe metadata or persisted its base64 payload");
    for (const std::string_view secret :
         {authorization_secret, cookie_secret, signed_query_secret}) {
        require(
            markdown.find(secret) == std::string::npos && jsonl.find(secret) == std::string::npos &&
                logs.find(secret) == std::string::npos && wire.find(secret) == std::string::npos,
            "HTTP credential or signed-query secret escaped a redacted boundary");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* scenario = std::getenv("NINFER_OPENAI_CACHE_SCENARIO");
    int devices          = 0;
    if (artifact == nullptr || *artifact == '\0' || cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices == 0) {
        std::cout << "skip: OpenAI cache regression requires Qwen3.8 weights and CUDA\n";
        return 77;
    }
    try {
        if (scenario != nullptr && std::string_view(scenario) == "engine-failure-shutdown") {
            exercise_http_engine_failure(artifact);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "reasoning-continuation") {
            exercise_generated_reasoning_continuation(artifact);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "concurrent-ingress-long") {
            exercise_nonblocking_durable_ingress(artifact, true);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "concurrent-ingress") {
            exercise_nonblocking_durable_ingress(artifact);
            exercise_changed_user_history(artifact);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "durable-two-lineage") {
            exercise_deferred_durable_deadline_settlement(artifact);
            exercise_durable_two_lineage_warm_reuse(artifact);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "ssd-memory-only-victim") {
            exercise_ssd_replaces_memory_only_checkpoint(artifact);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "ssd-side-request") {
            exercise_ssd_replacement_beside_background(artifact);
            return 0;
        }
        if (scenario != nullptr && std::string_view(scenario) == "service-ssd-replacement") {
            exercise_service_ssd_winning_replacement(artifact);
            return 0;
        }
        exercise_generated_reasoning_continuation(artifact);
        exercise_harness(artifact);
        exercise_disconnect_after_checkpoint_reuse(artifact);
        exercise_stream_failure_before_wait(artifact);
        exercise_deferred_durable_deadline_settlement(artifact);
        exercise_durable_two_lineage_warm_reuse(artifact);
        exercise_service_ssd_winning_replacement(artifact);
        exercise_ssd_replacement_beside_background(artifact);
        exercise_ssd_replaces_memory_only_checkpoint(artifact);
        exercise_protocol_content_capture(artifact);
        exercise_http_secret_exclusion(artifact);
        exercise_http_engine_failure(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
