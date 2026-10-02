// Real-artifact cross-lane isolation regression test.
//
// Single-sequence kernels (every prefill chunk) read their KV page-table row from one
// Program-wide device scalar. If a prefill chunk runs with another lane's row selected, the lane
// attends to that lane's pages or to stale pages left by earlier sessions, and its reply drifts
// toward an unrelated task. The other real generation tests submit one prompt on every lane, so
// they cannot observe this.
//
// This test first runs every prompt solo, which leaves retained sessions behind exactly as a
// server does, and then starts unrelated prompts together in several orders. A reply fails when
// it is empty or contains words that belong only to another prompt's task. Exact equality with
// the solo reply is deliberately not required: batch position may flip a near-tied token.
#include "ninfer/engine.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct Sink : ninfer::OutputSink {
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta) override {}
};

struct Case {
    std::string name;
    std::string text;
    // Words that appear in a correct reply to this prompt only.
    std::vector<std::string> own;
    // Words that belong to the other prompts' tasks and must never appear here.
    std::vector<std::string> forbidden;
    std::vector<ninfer::TokenId> tokens;
};

std::string vault_prompt(const char* word, const char* tail) {
    std::string text = "Facility log. Every entry below is authoritative.\n";
    for (int i = 0; i < 55; ++i) {
        text += "Entry " + std::to_string(i) + ": the vault code for room " + std::to_string(i) +
                " is " + word + "-" + std::to_string(i) + "-" + tail + ".\n";
    }
    text += "List the vault codes for rooms 7 to 40 exactly, one per line:";
    return text;
}

ninfer::RequestOptions greedy() {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 300;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 42;
    request.stop.include_model_defaults       = true;
    request.output.raw                        = true;
    return request;
}

bool contains(const std::string& text, const std::string& word) {
    return text.find(word) != std::string::npos;
}

int run(const char* artifact, bool mtp) {
    ninfer::EngineOptions options;
    options.artifact_path      = artifact;
    options.max_context        = 4096;
    options.kv_capacity        = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.max_concurrency    = 2;
    options.pending_timeout_ms = 300000;
    options.prefill_chunk      = 256;
    options.generation_reservation_tokens                   = 64;
    options.context_cache.host_kv_capacity_bytes            = 2ULL << 30;
    options.context_cache.host_state_slots                  = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.kv_cache                 = ninfer::KvCacheStorage::RK4V4E8;
    }
    ninfer::Engine engine(options);

    std::vector<Case> cases(4);
    cases[0] = {"numbers",
                "Write the numbers from one to one thousand, separated by commas:",
                {"1, 2, 3"},
                {"ORCHID", "TULIPS", "alphabet"},
                {}};
    cases[1] = {"vault",
                vault_prompt("ORCHID", "ZEBRA"),
                {"ORCHID"},
                {"TULIPS", "thousand", "alphabet"},
                {}};
    cases[2] = {"alphabet",
                "List the letters of the English alphabet in reverse order, one per line, with a "
                "word for each:",
                {"Zebra", "Z"},
                {"ORCHID", "TULIPS", "thousand"},
                {}};
    cases[3] = {"vault2",
                vault_prompt("TULIPS", "LLAMAS"),
                {"TULIPS"},
                {"ORCHID", "thousand", "alphabet"},
                {}};
    for (auto& item : cases) { item.tokens = engine.tokenize_text(item.text); }
    const auto request = greedy();

    // Solo runs leave retained sessions and recycled KV pages behind.
    for (auto& item : cases) {
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto result = engine.generate(engine.prepare_tokens(item.tokens), request);
            require(result.generated_token_ids.size() >= 20, item.name + " solo reply too short");
            for (const auto& word : item.forbidden) {
                require(!contains(result.content, word),
                        item.name + " solo reply contains foreign word '" + word + "'");
            }
        }
    }

    struct Pair {
        std::size_t first, second;
    };
    const std::vector<Pair> pairs = {{1, 3}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 0},
                                     {0, 1}, {3, 2}, {2, 3}, {1, 3}, {3, 1}, {2, 0}};
    int failures = 0;
    for (const auto& pair : pairs) {
        const std::size_t index[2] = {pair.first, pair.second};
        ninfer::GenerationHandle handles[2];
        std::vector<std::future<ninfer::GenerationResult>> results;
        std::vector<Sink> sinks(2);
        for (int slot = 0; slot < 2; ++slot) {
            handles[slot] = engine.submit(engine.prepare_tokens(cases[index[slot]].tokens), request,
                                          ninfer::OutputConsumerMode::Streaming);
        }
        for (int slot = 0; slot < 2; ++slot) {
            results.push_back(std::async(std::launch::async, [&, slot] {
                return handles[slot].wait(&sinks[slot]);
            }));
        }
        for (int slot = 0; slot < 2; ++slot) {
            const auto result = results[slot].get();
            const auto& item  = cases[index[slot]];
            std::string problem;
            if (result.generated_token_ids.size() < 20) { problem = "reply too short"; }
            for (const auto& word : item.forbidden) {
                if (contains(result.content, word)) { problem = "foreign word '" + word + "'"; }
            }
            bool has_own = item.own.empty();
            for (const auto& word : item.own) { has_own = has_own || contains(result.content, word); }
            if (!has_own) { problem = "reply lacks its own task words"; }
            if (!problem.empty()) {
                ++failures;
                std::cout << "LEAK pair=" << cases[index[0]].name << "+" << cases[index[1]].name
                          << " submitted_" << (slot == 0 ? "first" : "second") << " case="
                          << item.name << ": " << problem
                          << "\n    reply: " << result.content.substr(0, 240) << std::endl;
            }
        }
    }
    std::cout << "lane isolation mtp=" << mtp << ": " << (failures == 0 ? "PASS" : "FAIL") << " ("
              << failures << " bad replies in " << pairs.size() * 2 << ")" << std::endl;
    return failures == 0 ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (!artifact) {
        std::cout << "SKIP: Qwen3.8 artifact not configured\n";
        return 77;
    }
    try {
        return run(artifact, argc >= 2 && std::string_view(argv[1]) == "--mtp");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
