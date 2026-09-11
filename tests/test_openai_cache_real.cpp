#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::serve;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

RuntimeStats settled_stats(const GenerationService& service) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto stats = service.runtime_stats();
        if (stats.running_requests == 0 && stats.waiting_requests == 0 &&
            stats.materializing_requests == 0) {
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
                                   bool explicit_only = false) {
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
    // Exercise the production merge of protocol markers/policy with Responses retention hints.
    auto prepared = service.prepare(generation, GenerationConsumerMode::Aggregate, {}, hints);
    auto result   = service.run(prepared, nullptr);
    (void)settled_stats(service);
    require(!result.generated_token_ids.empty(), "OpenAI cache fixture generated no tokens");
    return result;
}

GenerationOutcome generate(GenerationService& service, const std::string& instructions,
                           const std::string& user, bool responses, bool explicit_only = false) {
    return generate_history(service,
                            Json::array({Json{{"role", "system"}, {"content", instructions}},
                                         Json{{"role", "user"}, {"content", user}}}),
                            responses, explicit_only);
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
                  << " state_h2d=" << after_continue.state_h2d_count - before_continue.state_h2d_count
                  << '\n';
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

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    int devices          = 0;
    if (artifact == nullptr || *artifact == '\0' || cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices == 0) {
        std::cout << "skip: OpenAI cache regression requires Qwen3.8 weights and CUDA\n";
        return 77;
    }
    try {
        exercise_harness(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
