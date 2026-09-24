#include "ninfer/engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
ninfer::PromptInput prompt(unsigned session, unsigned turn) {
    ninfer::PromptInput input;
    input.options.enable_thinking                 = false;
    input.context_cache.session_key               = "fork-pressure-" + std::to_string(session);
    input.context_cache.automatic_private_anchors = 2;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::System,
         .parts = {{.text = "Give precise, brief answers about programming."}}});
    for (unsigned i = 0; i <= turn; ++i) {
        input.messages.push_back(
            {.role  = ninfer::ChatRole::User,
             .parts = {{.text = "Explain scheduling for task " + std::to_string(session) +
                                " round " + std::to_string(i) + "."}}});
        if (i < turn) {
            input.messages.push_back({.role  = ninfer::ChatRole::Assistant,
                                      .parts = {{.text = "Run each ready task in order."}}});
        }
    }
    input.context_cache.markers.push_back({.after_message_count = 1});
    return input;
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (!artifact) { return 77; }
    try {
        ninfer::EngineOptions options;
        options.artifact_path            = artifact;
        options.max_context              = 4096;
        options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        options.max_concurrency          = 3;
        options.prefill_chunk            = 128;
        options.kv_cache                 = ninfer::KvCacheStorage::RK4V4E8;
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.context_cache.device_state_slots                = 3;
        options.context_cache.host_state_slots                  = 16;
        options.context_cache.host_kv_capacity_bytes            = 512ULL << 20;
        options.context_cache.max_private_continuations         = 6;
        options.context_cache.max_shared_prefixes               = 4;
        options.context_cache.max_long_anchors_per_continuation = 2;
        ninfer::Engine engine(options);
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 64;
        request.execution.sampling.temperature    = 0;
        request.stop.include_model_defaults       = false;
        for (unsigned turn = 0; turn < 8; ++turn) {
            std::vector<ninfer::GenerationHandle> handles;
            for (unsigned session = 0; session < 3; ++session) {
                handles.push_back(engine.submit(engine.prepare(prompt(session, turn)), request));
            }
            for (auto& handle : handles) {
                const auto result = handle.wait();
                if (result.generated_token_ids.size() != 64) {
                    throw std::runtime_error("concurrent cached request did not complete");
                }
            }
            std::cout << "turn=" << turn << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
