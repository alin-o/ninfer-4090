// Real-artifact coverage for two additions to the public Engine route:
//
//   1. Engine::prepare_tokens(token_ids, allow_prefix_identity, shared_prefix_frontiers) --
//      opt-in SharedStablePrefix cache markers for sibling raw-token prompts that share a
//      prefix but diverge in suffix. A frontier declared by an *establishing* request is
//      captured into the shared catalog (shared_candidates / shared_catalog_) once that
//      request publishes a continuation; any later sibling request whose own prefix digest
//      matches at that frontier reuses it, even without redeclaring the frontier itself (the
//      consuming side matches via its own whole-prompt PrefixShortlistDigests rolling digest,
//      queried at the catalog entry's registered frontier -- see prefix_identity.h).
//   2. Engine::slot_tokens(lane, expected_digest) -- a pure read of an idle retained slot's
//      confirmed-reusable token ledger (the sequence's execution_frontier, which excludes a
//      just-generated trailing token not yet confirmed by a forward pass).
//
// Uses SpeculativeBackend::Mtp / ProposalHead::Optimized throughout, matching the backend
// production ninfer-serve runs with (see test_engine_score_candidates_real.cpp's fixture).
// Candidate-score and continuation outputs are cross-checked against the independent
// CausalScoring Engine route (Engine::score_tokens), built on the same artifact only after
// every Generation Engine above is destroyed (the 27B target cannot be resident twice at once).
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::EngineOptions mtp_generation_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 2;
    // The default private catalog capacity is 2*max_concurrency (=2 here): far too small for
    // this single-threaded test's actual working set of 5 live private continuations
    // (establishing, sibling A, sibling B, the impostor guard, and the Task 2 retained slot).
    // Private retention class is a soft eviction-cost input to the materialization planner's
    // portfolio search, not a fixed protection order (see docs/maintainer/resource-scheduling-and-
    // context-cache.md); a never-yet-reused freshly retained slot can legitimately be judged the
    // *cheapest* eviction victim and lose to an unrelated fresh request even though the matching
    // candidate itself offers far more token reuse. That is a real planner trade-off, not a
    // scoring-path defect -- avoid depending on winning that race here by giving every live
    // continuation plus the suffix-extended scoring fork its own catalog cell, so the identity
    // fast path (an outright free cell) is always available and no eviction choice is needed.
    options.context_cache.max_private_continuations = 6;
    return options;
}

std::vector<ninfer::TokenId> repeated_prompt(std::size_t length, ninfer::TokenId base) {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        tokens.push_back(base + static_cast<ninfer::TokenId>(i % 97));
    }
    return tokens;
}

ninfer::RequestOptions greedy_continuation_request(std::uint32_t output_tokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = output_tokens;
    request.execution.sampling.temperature     = 0.0F;
    request.execution.allow_prefix_reuse       = true;
    request.stop.include_model_defaults        = false;
    return request;
}

ninfer::RequestOptions scoring_request(std::vector<ninfer::TokenId> candidates) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature     = 0.0F;
    request.execution.allow_prefix_reuse       = true;
    request.execution.score_candidates         = std::move(candidates);
    request.stop.include_model_defaults        = false;
    return request;
}

// prepare_tokens frontier-validation rejections. These are cheap: no generation is submitted, so
// they are exercised before any GPU work to keep the failure-mode coverage independent of the
// artifact's actual content.
int exercise_frontier_rejections(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    const auto expect_invalid = [&](std::vector<std::uint32_t> frontiers, const char* label) {
        try {
            auto prepared = engine.prepare_tokens(prompt, true, frontiers);
            (void)prepared;
            std::cerr << "expected shared_prefix_frontiers rejection did not occur: " << label
                      << '\n';
            return 1;
        } catch (const std::invalid_argument&) {
            return 0;
        }
    };

    if (const int rc = expect_invalid({0}, "frontier == 0"); rc != 0) { return rc; }
    if (const int rc = expect_invalid({static_cast<std::uint32_t>(prompt.size())},
                                      "frontier == prompt.size()");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_invalid({static_cast<std::uint32_t>(prompt.size() + 10)},
                                      "frontier > prompt.size()");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_invalid({200, 100}, "non-ascending frontiers"); rc != 0) {
        return rc;
    }
    if (const int rc = expect_invalid({100, 100}, "duplicate (non-strictly-ascending) frontiers");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_invalid({50, 100, 150, 200, 250}, "> 4 frontiers"); rc != 0) {
        return rc;
    }

    // Exactly 4 frontiers, strictly ascending and strictly inside the prompt, is accepted.
    try {
        const std::vector<std::uint32_t> four_frontiers{50, 100, 150, 200};
        auto prepared = engine.prepare_tokens(prompt, true, four_frontiers);
        (void)prepared;
    } catch (const std::exception& error) {
        std::cerr << "4 valid ascending frontiers were rejected: " << error.what() << '\n';
        return 1;
    }

    // The plain two-argument call (no frontiers) must keep working unchanged.
    try {
        auto prepared = engine.prepare_tokens(prompt, true);
        (void)prepared;
    } catch (const std::exception& error) {
        std::cerr << "plain two-argument prepare_tokens call was rejected: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    const std::vector<ninfer::TokenId> shared_prefix = repeated_prompt(600, 300);
    const std::uint32_t frontier                     = static_cast<std::uint32_t>(shared_prefix.size());
    const std::vector<ninfer::TokenId> candidates{1000, 2000, 3000, 4000, 5000};

    std::vector<ninfer::TokenId> establishing_prompt = shared_prefix;
    establishing_prompt.push_back(7001); // establishing request's own suffix

    std::vector<ninfer::TokenId> sibling_a_prompt = shared_prefix;
    sibling_a_prompt.push_back(7002);
    std::vector<ninfer::TokenId> sibling_b_prompt = shared_prefix;
    sibling_b_prompt.insert(sibling_b_prompt.end(), {7003, 7004, 7005});

    // False-positive guard: differs in the FIRST token of the "shared" prefix, same overall
    // length as establishing_prompt. This must never be treated as reusing the shared prefix.
    std::vector<ninfer::TokenId> impostor_prompt = shared_prefix;
    impostor_prompt[0] += 1;
    impostor_prompt.push_back(7001);

    std::vector<ninfer::TokenId> generated_a;
    std::vector<ninfer::TokenId> generated_b;
    std::vector<float> candidate_scores_sibling;
    std::uint32_t sibling_a_reused = 0;
    std::uint32_t sibling_b_reused = 0;
    std::uint32_t impostor_reused  = 0;

    std::vector<ninfer::TokenId> slot_round_trip_prompt;
    std::vector<ninfer::TokenId> slot_tokens_result;
    std::vector<ninfer::TokenId> slot_suffix_score_prompt;
    std::vector<float> slot_suffix_scores;
    std::uint32_t slot_suffix_reused = 0;
    bool wrong_digest_rejected       = false;
    bool erased_slot_rejected        = false;

    {
        ninfer::Engine engine(mtp_generation_engine_options(artifact));

        if (const int rc = exercise_frontier_rejections(engine, shared_prefix); rc != 0) {
            return rc;
        }

        // The very first (establishing) request must not see any reuse benefit: nothing has been
        // published to match against yet.
        const std::vector<std::uint32_t> establishing_frontiers{frontier};
        const ninfer::GenerationResult establishing = engine.generate(
            engine.prepare_tokens(establishing_prompt, true, establishing_frontiers),
            greedy_continuation_request(1));
        if (establishing.reused_prompt_tokens != 0) {
            std::cerr << "establishing request unexpectedly reused "
                      << establishing.reused_prompt_tokens << " tokens before anything was published\n";
            return 1;
        }

        // Sibling A: diverges right after the shared frontier. Must reuse the full shared prefix
        // even though it never declared the frontier itself.
        const ninfer::GenerationResult a =
            engine.generate(engine.prepare_tokens(sibling_a_prompt), greedy_continuation_request(4));
        if (a.reused_prompt_tokens != shared_prefix.size()) {
            std::cerr << "sibling A reused " << a.reused_prompt_tokens << " tokens, expected the full "
                        "shared prefix of "
                      << shared_prefix.size() << '\n';
            return 1;
        }
        sibling_a_reused = a.reused_prompt_tokens;
        generated_a      = a.generated_token_ids;

        // Sibling B: a second, independently diverging sibling. Proves sibling A did not
        // consume/advance the shared catalog entry out from under it.
        const ninfer::GenerationResult b =
            engine.generate(engine.prepare_tokens(sibling_b_prompt), greedy_continuation_request(4));
        if (b.reused_prompt_tokens != shared_prefix.size()) {
            std::cerr << "sibling B reused " << b.reused_prompt_tokens << " tokens, expected the full "
                        "shared prefix of "
                      << shared_prefix.size()
                      << " (sibling A must not have consumed the shared catalog entry)\n";
            return 1;
        }
        sibling_b_reused = b.reused_prompt_tokens;
        generated_b      = b.generated_token_ids;

        // A scoring request against the same shared prefix also reuses it correctly.
        std::vector<ninfer::TokenId> sibling_c_prompt = shared_prefix;
        sibling_c_prompt.push_back(7006);
        const ninfer::GenerationResult c =
            engine.generate(engine.prepare_tokens(sibling_c_prompt), scoring_request(candidates));
        if (c.reused_prompt_tokens != shared_prefix.size()) {
            std::cerr << "scoring sibling reused " << c.reused_prompt_tokens
                      << " tokens, expected the full shared prefix of " << shared_prefix.size() << '\n';
            return 1;
        }
        candidate_scores_sibling = c.candidate_logprobs;

        // False positive: a prompt that differs in the first token of the "shared" region, same
        // length as establishing_prompt, must not be treated as a shared-prefix reuse.
        const ninfer::GenerationResult impostor = engine.generate(
            engine.prepare_tokens(impostor_prompt), greedy_continuation_request(1));
        impostor_reused = impostor.reused_prompt_tokens;
        if (impostor_reused >= frontier) {
            std::cerr << "impostor prompt (differing first token) incorrectly reused "
                      << impostor_reused << " tokens (>= frontier " << frontier << ")\n";
            return 1;
        }

        // --- Task 2: slot_tokens round trip ---
        // A fresh ordinary generation request, retained to a slot.
        slot_round_trip_prompt = repeated_prompt(300, 500);
        const ninfer::GenerationResult retained =
            engine.generate(engine.prepare_tokens(slot_round_trip_prompt), greedy_continuation_request(1));
        if (retained.slot < 0 || retained.session_digest.empty()) {
            std::cerr << "slot_tokens fixture generation did not retain a catalogued session\n";
            return 1;
        }
        const auto slot = static_cast<std::uint32_t>(retained.slot);

        slot_tokens_result = engine.slot_tokens(slot, retained.session_digest);
        // execution_frontier excludes the one just-generated, not-yet-confirmed trailing token:
        // it must equal the original prompt exactly, not prompt+generated_token.
        if (slot_tokens_result != slot_round_trip_prompt) {
            std::cerr << "slot_tokens returned " << slot_tokens_result.size()
                      << " tokens, expected exactly the original prompt's " << slot_round_trip_prompt.size()
                      << " tokens (execution_frontier excludes the unconfirmed trailing token)\n";
            return 1;
        }

        // Exception contract: a non-empty wrong digest raises SlotSessionMismatch.
        try {
            auto wrong = engine.slot_tokens(slot, "0000000000000000");
            (void)wrong;
            std::cerr << "slot_tokens accepted a wrong expected_digest\n";
            return 1;
        } catch (const ninfer::SlotSessionMismatch&) {
            wrong_digest_rejected = true;
        }

        // Use slot_tokens's result to build a suffix-extended prompt and verify it reuses the full
        // slot_tokens length via an ordinary score_candidates request against the same slot's
        // session (round-tripping Task 2's output back through Task 1's raw-token reuse path).
        slot_suffix_score_prompt = slot_tokens_result;
        const std::vector<ninfer::TokenId> extra = repeated_prompt(20, 8000);
        slot_suffix_score_prompt.insert(slot_suffix_score_prompt.end(), extra.begin(), extra.end());
        const ninfer::GenerationResult suffix_scored = engine.generate(
            engine.prepare_tokens(slot_suffix_score_prompt), scoring_request(candidates));
        if (suffix_scored.reused_prompt_tokens != slot_tokens_result.size()) {
            std::cerr << "suffix-extended slot_tokens prompt reused " << suffix_scored.reused_prompt_tokens
                      << " tokens, expected exactly slot_tokens's " << slot_tokens_result.size() << '\n';
            return 1;
        }
        slot_suffix_reused = suffix_scored.reused_prompt_tokens;
        slot_suffix_scores = suffix_scored.candidate_logprobs;

        // Exception contract: erase_slot empties the lane, then slot_tokens on it raises
        // std::invalid_argument. (The "busy slot -> RequestError(Overloaded)" case cannot be
        // exercised in this single-threaded test: it requires a concurrent request actively
        // holding the lane, which is left as a documented, unverified limitation.)
        (void)engine.erase_slot(slot);
        try {
            auto erased = engine.slot_tokens(slot);
            (void)erased;
            std::cerr << "slot_tokens accepted an empty/erased slot\n";
            return 1;
        } catch (const std::invalid_argument&) {
            erased_slot_rejected = true;
        }
    } // Mtp Generation Engine destroyed here, freeing the 27B residency for the oracle.

    // Independent CausalScoring oracle, cross-checking the candidate scores obtained via
    // shared-prefix reuse. Scoped on its own: the 27B target cannot be resident twice at once, and
    // the fresh-Mtp-Engine greedy-continuation cross-check below needs its own full residency after
    // this oracle is torn down.
    {
        ninfer::EngineOptions oracle_options;
        oracle_options.artifact_path = artifact;
        oracle_options.purpose       = ninfer::EnginePurpose::CausalScoring;
        oracle_options.max_context   = 4096;
        ninfer::Engine oracle(std::move(oracle_options));

        const auto oracle_scores = [&](const std::vector<ninfer::TokenId>& base_prompt) {
            std::vector<float> scores;
            scores.reserve(candidates.size());
            for (const ninfer::TokenId candidate : candidates) {
                std::vector<ninfer::TokenId> window = base_prompt;
                window.push_back(candidate);
                const std::vector<float> result =
                    oracle.score_tokens(window, static_cast<std::uint32_t>(base_prompt.size()));
                if (result.size() != 1) {
                    throw std::logic_error("causal scoring oracle returned an unexpected shape");
                }
                scores.push_back(result.front());
            }
            return scores;
        };

        // A forked-through-quantized-KV path (shared-prefix reuse, or slot_tokens's retained
        // private continuation) recomputes its suffix on top of a KV prefix that was stored and
        // read back through the engine's cache-type-k/v quantization, not kept in the oracle's
        // higher-precision computation. This is an expected consequence of the storage boundary,
        // not a reuse-correctness bug, so reused-path comparisons get their own, still tightly
        // bounded, tolerance (see test_engine_score_candidates_real.cpp's identical precedent).
        constexpr float kToleranceReused = 0.1F;
        const auto compare = [&](const std::vector<float>& engine_scores,
                                 const std::vector<float>& reference, const char* label,
                                 float tolerance) {
            float max_error = 0.0F;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                max_error = std::max(max_error, std::abs(engine_scores[i] - reference[i]));
            }
            if (max_error > tolerance) {
                std::cerr << label << " diverged from the CausalScoring oracle by " << max_error
                          << " (tolerance " << tolerance << ")\n";
                return 1;
            }
            return 0;
        };

        std::vector<ninfer::TokenId> sibling_c_prompt = shared_prefix;
        sibling_c_prompt.push_back(7006);
        const std::vector<float> sibling_c_reference = oracle_scores(sibling_c_prompt);
        if (const int rc = compare(candidate_scores_sibling, sibling_c_reference,
                                   "shared-prefix sibling scoring", kToleranceReused);
            rc != 0) {
            return rc;
        }

        const std::vector<float> suffix_reference = oracle_scores(slot_suffix_score_prompt);
        if (const int rc = compare(slot_suffix_scores, suffix_reference,
                                   "slot_tokens suffix-extended scoring", kToleranceReused);
            rc != 0) {
            return rc;
        }
    } // CausalScoring oracle destroyed here, freeing the 27B residency for the fresh Mtp Engine.

    // Greedy continuations: a fresh Mtp Engine re-decoding the same prompts from scratch must
    // produce identical tokens, proving shared-prefix reuse did not perturb decode state.
    {
        ninfer::Engine fresh_engine(mtp_generation_engine_options(artifact));
        const ninfer::GenerationResult reference_a = fresh_engine.generate(
            fresh_engine.prepare_tokens(sibling_a_prompt), greedy_continuation_request(4));
        if (reference_a.generated_token_ids != generated_a) {
            std::cerr << "sibling A's shared-prefix-reuse decode diverged from a fresh Engine\n";
            return 1;
        }
        const ninfer::GenerationResult reference_b = fresh_engine.generate(
            fresh_engine.prepare_tokens(sibling_b_prompt), greedy_continuation_request(4));
        if (reference_b.generated_token_ids != generated_b) {
            std::cerr << "sibling B's shared-prefix-reuse decode diverged from a fresh Engine\n";
            return 1;
        }
    }

    if (!wrong_digest_rejected || !erased_slot_rejected) {
        std::cerr << "slot_tokens exception-contract checks did not all run\n";
        return 1;
    }

    std::cout << "OK engine_shared_prefix_tokens_real (sibling_a_reused=" << sibling_a_reused
              << " sibling_b_reused=" << sibling_b_reused << " impostor_reused=" << impostor_reused
              << " slot_tokens=" << slot_tokens_result.size()
              << " slot_suffix_reused=" << slot_suffix_reused << ")\n";
    return 0;
}
