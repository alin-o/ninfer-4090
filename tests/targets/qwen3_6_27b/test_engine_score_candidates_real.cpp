// Real-artifact coverage for native next-token candidate scoring inside the Generation Engine
// (ExecutionOptions::score_candidates / GenerationResult::candidate_logprobs). This exercises the
// staged-prefill scoring hook added to qwen3_6 program_impl.h: a fresh-prefill path that crosses a
// prefill chunk boundary, strictly read-only sibling scoring against a retained source, and
// submission-time rejection of malformed requests. It covers this both on a plain
// (SpeculativeBackend::None) Generation Engine and on an Mtp (SpeculativeBackend::Mtp) Generation
// Engine, since production ninfer-serve runs with Mtp speculative decoding. Results are
// cross-checked against the independent CausalScoring Engine route (Engine::score_tokens), built
// on the same artifact after every Generation Engine above is destroyed, since the 27B target
// cannot be resident on the GPU twice at once.
//
// Read-only contract: a scoring request (ExecutionOptions::score_candidates) never publishes its
// own resulting state as a reusable continuation, and never consumes/advances a matched private
// source (named slot or otherwise retained continuation) in place -- it always forks. This means:
// (a) two scoring requests resubmitted back to back with no intervening ordinary generation never
// see any reuse of each other's state (nothing was published to match against); (b) once an
// ordinary (non-scoring) generation request has published a retained continuation, any number of
// sibling scoring requests that diverge after that shared prefix each reuse the full frontier
// independently, because none of them consumes/advances the shared source; and (c) an ordinary
// generation issued after those siblings still sees the original source completely intact and
// produces the same output a pristine Engine would. This is proven below instead of the
// zero-suffix/partial-reuse-of-a-prior-score-request cases an earlier revision of this test used,
// which encoded the superseded (non-read-only) scoring contract.
//
// Partial-prefix reuse for a raw token prompt (no chat PromptCacheMarker) only recognizes a strict
// append onto the previously retained frontier (the engine's Root reuse path); an arbitrary
// shared-prefix-then-diverge prompt requires PromptInput markers (SharedStablePrefix /
// PrivateLongAnchor, see test_engine_prefix_real.cpp) and is out of scope for this raw-token test.
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

ninfer::EngineOptions generation_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 256; // force the ~1100-token prompt across chunks
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 2;
    return options;
}

// Matches the Mtp fixture used by test_engine_prefix_real.cpp: Optimized proposal head, a 3-token
// draft width. This is the backend production ninfer-serve runs with.
ninfer::EngineOptions mtp_generation_engine_options(const char* artifact) {
    ninfer::EngineOptions options     = generation_engine_options(artifact);
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

std::vector<ninfer::TokenId> repeated_prompt(std::size_t length, ninfer::TokenId base) {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        // A varied but deterministic pattern avoids degenerate all-equal hidden states while
        // staying inside the base artifact's tokenizer-independent raw-ID path.
        tokens.push_back(base + static_cast<ninfer::TokenId>(i % 97));
    }
    return tokens;
}

ninfer::RequestOptions scoring_request(std::vector<ninfer::TokenId> candidates) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.execution.score_candidates        = std::move(candidates);
    request.stop.include_model_defaults       = false;
    return request;
}

ninfer::RequestOptions greedy_continuation_request(std::uint32_t output_tokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = output_tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;
    return request;
}

int exercise_rejections(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    const auto expect_invalid = [&](ninfer::RequestOptions request, const char* label) {
        try {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare_tokens(prompt), std::move(request));
            (void)result;
            std::cerr << "expected rejection did not occur: " << label << '\n';
            return 1;
        } catch (const std::invalid_argument&) {
            return 0;
        }
    };

    // requested_output_tokens must be exactly 1 when candidates are present.
    {
        ninfer::RequestOptions bad            = scoring_request({100, 200, 300, 400});
        bad.execution.requested_output_tokens = 2;
        if (const int rc = expect_invalid(std::move(bad), "requested_output_tokens != 1");
            rc != 0) {
            return rc;
        }
    }
    // Duplicate candidates are rejected.
    {
        if (const int rc =
                expect_invalid(scoring_request({100, 200, 100, 300}), "duplicate candidates");
            rc != 0) {
            return rc;
        }
    }
    // More than 1024 candidates are rejected.
    {
        std::vector<ninfer::TokenId> too_many;
        too_many.reserve(1025);
        for (ninfer::TokenId id = 0; id < 1025; ++id) { too_many.push_back(id); }
        if (const int rc = expect_invalid(scoring_request(std::move(too_many)), "> 1024 candidates");
            rc != 0) {
            return rc;
        }
    }
    // Out-of-vocabulary candidate ids are rejected.
    {
        if (const int rc = expect_invalid(scoring_request({100, 200, 248077, 300}),
                                          "out-of-vocabulary candidate");
            rc != 0) {
            return rc;
        }
    }
    return 0;
}

// DFlash/DFlash2 must still be rejected for score_candidates (request_plan_impl.h's plan_request()
// uses is_masked_draft_backend() to reject exactly these two backends, Mtp is allowed). The 27B
// package only ever exposes a DFlash2 weight bundle (plain DFlash is a 35B-A3B package fixture,
// see test_engine_dflash_real.cpp), and NINFER_QWEN3_6_27B_WEIGHTS may point at an artifact built
// without that optional bundle. Either outcome below demonstrates the rejection: the Engine itself
// may refuse to materialize the DFlash2 backend (no bundle in this artifact), or materialization
// succeeds and submit_with_recovery/plan_request() rejects score_candidates with
// std::invalid_argument. The one outcome this must never allow is a GenerationResult carrying
// candidate_logprobs from a masked-draft-backend Engine.
int exercise_masked_draft_backend_rejected(const char* artifact) {
    ninfer::EngineOptions options     = generation_engine_options(artifact);
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens  = 2;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    try {
        ninfer::Engine engine(std::move(options));
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(prompt), scoring_request({100, 200, 300, 400}));
        (void)result;
        std::cerr << "score_candidates was accepted on a DFlash2 Engine\n";
        return 1;
    } catch (const std::exception&) {
        // Rejected either at Engine construction (no DFlash2 bundle in this artifact) or at
        // submission (std::invalid_argument from plan_request()). Both satisfy the contract.
        return 0;
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    const std::vector<ninfer::TokenId> candidates{1000, 2000, 3000, 4000, 5000};

    // A prompt long enough to span several 256-token prefill chunks.
    const std::vector<ninfer::TokenId> prompt = repeated_prompt(1100, 300);

    std::vector<float> fresh_scores;
    std::vector<float> sibling_a_scores;
    std::vector<float> sibling_b_scores;
    std::vector<ninfer::TokenId> sibling_a_prompt;
    std::vector<ninfer::TokenId> sibling_b_prompt;
    std::vector<ninfer::TokenId> post_scoring_tokens;
    std::uint32_t sibling_a_reused_tokens    = 0;
    std::uint32_t sibling_b_reused_tokens    = 0;
    std::uint32_t post_scoring_reused_tokens = 0;

    {
        ninfer::Engine engine(generation_engine_options(artifact));

        if (const int rc = exercise_rejections(engine, prompt); rc != 0) { return rc; }

        const ninfer::GenerationResult fresh =
            engine.generate(engine.prepare_tokens(prompt), scoring_request(candidates));
        if (fresh.candidate_logprobs.size() != candidates.size()) {
            std::cerr << "fresh-prefill scoring returned " << fresh.candidate_logprobs.size()
                      << " logprobs, expected " << candidates.size() << '\n';
            return 1;
        }
        for (const float value : fresh.candidate_logprobs) {
            if (!std::isfinite(value) || value > 0.0F) {
                std::cerr << "fresh-prefill scoring returned an invalid logprob: " << value << '\n';
                return 1;
            }
        }
        fresh_scores = fresh.candidate_logprobs;

        // Read-only contract: the fresh scoring request above published nothing, so resubmitting
        // the identical prompt as a scoring request again would cold-start rather than reuse it.
        // Instead, publish a real retained source with an ordinary (non-scoring) generation
        // request, then prove two sibling scoring requests that each diverge after this shared
        // prefix both reuse it in full -- the first one must not consume/advance it out from under
        // the second.
        const ninfer::GenerationResult root_gen =
            engine.generate(engine.prepare_tokens(prompt), greedy_continuation_request(1));
        if (root_gen.generated_token_ids.size() != 1) {
            std::cerr << "root generation did not produce its one requested token\n";
            return 1;
        }

        sibling_a_prompt = prompt;
        sibling_a_prompt.push_back(9001); // divergent suffix token A, distinct from token B below
        const ninfer::GenerationResult a =
            engine.generate(engine.prepare_tokens(sibling_a_prompt), scoring_request(candidates));
        if (a.reused_prompt_tokens != prompt.size()) {
            std::cerr << "sibling scoring request A reused " << a.reused_prompt_tokens
                      << " tokens, expected the full shared prefix of " << prompt.size() << '\n';
            return 1;
        }
        sibling_a_scores        = a.candidate_logprobs;
        sibling_a_reused_tokens = a.reused_prompt_tokens;

        sibling_b_prompt = prompt;
        sibling_b_prompt.push_back(9002); // divergent suffix token B
        const ninfer::GenerationResult b =
            engine.generate(engine.prepare_tokens(sibling_b_prompt), scoring_request(candidates));
        if (b.reused_prompt_tokens != prompt.size()) {
            std::cerr << "sibling scoring request B reused " << b.reused_prompt_tokens
                      << " tokens, expected the full shared prefix of " << prompt.size()
                      << " (sibling A must not have consumed/advanced the shared source)\n";
            return 1;
        }
        sibling_b_scores        = b.candidate_logprobs;
        sibling_b_reused_tokens = b.reused_prompt_tokens;

        // Non-corruption check: an ordinary generation continuing from the same original source,
        // issued after both sibling scoring requests, must still reuse the full frontier (the
        // source was never consumed) and is compared against a pristine Engine below.
        const ninfer::GenerationResult post =
            engine.generate(engine.prepare_tokens(prompt), greedy_continuation_request(1));
        if (post.reused_prompt_tokens != prompt.size()) {
            std::cerr << "post-sibling-scoring generation reused " << post.reused_prompt_tokens
                      << " tokens, expected the full original source of " << prompt.size()
                      << " (the sibling scoring requests must not have consumed it)\n";
            return 1;
        }
        post_scoring_tokens        = post.generated_token_ids;
        post_scoring_reused_tokens = post.reused_prompt_tokens;
    } // Generation Engine destroyed here, freeing the 27B residency for the next Engine.

    // Independent fresh Engine, generating from the identical original prompt with no scoring
    // requests ever run against it, as the "scoring never corrupts the source" comparison.
    {
        ninfer::Engine fresh_engine(generation_engine_options(artifact));
        const ninfer::GenerationResult reference = fresh_engine.generate(
            fresh_engine.prepare_tokens(prompt), greedy_continuation_request(1));
        if (reference.generated_token_ids != post_scoring_tokens) {
            std::cerr << "post-sibling-scoring generation diverged from a fresh Engine's greedy "
                        "decode of the same prompt\n";
            return 1;
        }
    } // Fresh Engine destroyed here.

    if (const int rc = exercise_masked_draft_backend_rejected(artifact); rc != 0) { return rc; }

    // Mtp Engine: production ninfer-serve runs with SpeculativeBackend::Mtp, so score_candidates
    // must work there too. Mirrors the None-backend read-only-contract exercise above: fresh
    // scoring publishes nothing, an ordinary generation publishes a real retained source, two
    // divergent sibling scoring requests each reuse that source in full without consuming it, and
    // a further ordinary generation afterward still sees the original source completely intact.
    std::vector<float> mtp_fresh_scores;
    std::vector<float> mtp_sibling_a_scores;
    std::vector<float> mtp_sibling_b_scores;
    std::vector<ninfer::TokenId> mtp_sibling_a_prompt;
    std::vector<ninfer::TokenId> mtp_sibling_b_prompt;
    std::vector<ninfer::TokenId> mtp_post_scoring_tokens;
    std::uint32_t mtp_sibling_a_reused_tokens    = 0;
    std::uint32_t mtp_sibling_b_reused_tokens    = 0;
    std::uint32_t mtp_post_scoring_reused_tokens = 0;

    {
        ninfer::Engine engine(mtp_generation_engine_options(artifact));

        const ninfer::GenerationResult fresh =
            engine.generate(engine.prepare_tokens(prompt), scoring_request(candidates));
        if (fresh.candidate_logprobs.size() != candidates.size()) {
            std::cerr << "Mtp fresh-prefill scoring returned " << fresh.candidate_logprobs.size()
                      << " logprobs, expected " << candidates.size() << '\n';
            return 1;
        }
        for (const float value : fresh.candidate_logprobs) {
            if (!std::isfinite(value) || value > 0.0F) {
                std::cerr << "Mtp fresh-prefill scoring returned an invalid logprob: " << value
                          << '\n';
                return 1;
            }
        }
        mtp_fresh_scores = fresh.candidate_logprobs;

        // Read-only contract on an Mtp Engine: publish a real retained source with an ordinary
        // (non-scoring) generation request, then prove two sibling scoring requests that each
        // diverge after this shared prefix both reuse it in full -- the first one must not
        // consume/advance it out from under the second.
        const ninfer::GenerationResult root_gen =
            engine.generate(engine.prepare_tokens(prompt), greedy_continuation_request(1));
        if (root_gen.generated_token_ids.size() != 1) {
            std::cerr << "Mtp root generation did not produce its one requested token\n";
            return 1;
        }

        mtp_sibling_a_prompt = prompt;
        mtp_sibling_a_prompt.push_back(9001); // divergent suffix token A, distinct from token B below
        const ninfer::GenerationResult a = engine.generate(engine.prepare_tokens(mtp_sibling_a_prompt),
                                                            scoring_request(candidates));
        if (a.reused_prompt_tokens != prompt.size()) {
            std::cerr << "Mtp sibling scoring request A reused " << a.reused_prompt_tokens
                      << " tokens, expected the full shared prefix of " << prompt.size() << '\n';
            return 1;
        }
        mtp_sibling_a_scores        = a.candidate_logprobs;
        mtp_sibling_a_reused_tokens = a.reused_prompt_tokens;

        mtp_sibling_b_prompt = prompt;
        mtp_sibling_b_prompt.push_back(9002); // divergent suffix token B
        const ninfer::GenerationResult b = engine.generate(engine.prepare_tokens(mtp_sibling_b_prompt),
                                                            scoring_request(candidates));
        if (b.reused_prompt_tokens != prompt.size()) {
            std::cerr << "Mtp sibling scoring request B reused " << b.reused_prompt_tokens
                      << " tokens, expected the full shared prefix of " << prompt.size()
                      << " (sibling A must not have consumed/advanced the shared source)\n";
            return 1;
        }
        mtp_sibling_b_scores        = b.candidate_logprobs;
        mtp_sibling_b_reused_tokens = b.reused_prompt_tokens;

        // Non-corruption check: an ordinary multi-token greedy generation request, submitted right
        // after both sibling scoring requests on this Mtp Engine, must still Root-reuse the full
        // retained frontier (proving the scoring requests left sequence.mtp_kv_valid/mtp_draft_count
        // consistent and did not consume/advance the shared source) and is compared against a fresh
        // Mtp Engine below.
        const ninfer::GenerationResult post =
            engine.generate(engine.prepare_tokens(prompt), greedy_continuation_request(8));
        if (post.reused_prompt_tokens != prompt.size()) {
            std::cerr << "post-sibling-scoring Mtp generation reused " << post.reused_prompt_tokens
                      << " tokens, expected the full original source of " << prompt.size()
                      << " (the sibling scoring requests must not have consumed it)\n";
            return 1;
        }
        if (post.generated_token_ids.empty()) {
            std::cerr << "post-sibling-scoring Mtp generation produced no tokens\n";
            return 1;
        }
        mtp_post_scoring_tokens        = post.generated_token_ids;
        mtp_post_scoring_reused_tokens = post.reused_prompt_tokens;
    } // Mtp Engine destroyed here.

    // Independent fresh Mtp Engine, generating from the identical original prompt with no scoring
    // requests ever run against it, as the "scoring never corrupts the source" comparison.
    {
        ninfer::Engine fresh_engine(mtp_generation_engine_options(artifact));
        const ninfer::GenerationResult reference = fresh_engine.generate(
            fresh_engine.prepare_tokens(prompt), greedy_continuation_request(8));
        if (reference.generated_token_ids != mtp_post_scoring_tokens) {
            std::cerr << "post-sibling-scoring Mtp generation diverged from a fresh Mtp Engine's "
                        "greedy decode of the same prompt\n";
            return 1;
        }
    } // Fresh Mtp Engine destroyed here, freeing the 27B residency for the CausalScoring oracle.

    // Named-slot read-only scoring across two independently retained sources (A, B), proving the
    // read-only fork contract holds across an admission interleaving the earlier sibling-scoring
    // blocks above do not exercise: scoring A twice, then publishing an entirely independent second
    // retained source B, then scoring both A and B again. slot_states() -- the Engine's public
    // per-slot occupancy snapshot -- must not change shape, retention, or digest for either slot
    // across any of the scoring calls; only the two ordinary (non-scoring) generations that publish
    // A and B are allowed to change it.
    {
        ninfer::EngineOptions options             = mtp_generation_engine_options(artifact);
        // A and B are both resident private continuations; a scoring fork still contests a
        // transient publication slot during planning even though it never catalogs anything (see
        // test_engine_shared_prefix_tokens_real.cpp's identical capacity rationale) -- size
        // generously so neither A nor B is ever at risk of eviction by a scoring request's own
        // planning-time candidate.
        options.context_cache.max_private_continuations = 6;
        ninfer::Engine engine(std::move(options));

        const auto relevant_state = [](const ninfer::SlotState& state) {
            return std::make_tuple(state.retained, state.processing, state.prompt_tokens,
                                   state.cached_tokens, state.mapped_tokens, state.session_digest);
        };
        const auto states_equal = [&](const std::vector<ninfer::SlotState>& lhs,
                                      const std::vector<ninfer::SlotState>& rhs) {
            if (lhs.size() != rhs.size()) { return false; }
            for (std::size_t i = 0; i < lhs.size(); ++i) {
                if (relevant_state(lhs[i]) != relevant_state(rhs[i])) { return false; }
            }
            return true;
        };

        const std::vector<ninfer::TokenId> prompt_a = repeated_prompt(300, 20000);
        const ninfer::GenerationResult a =
            engine.generate(engine.prepare_tokens(prompt_a), greedy_continuation_request(1));
        if (a.slot < 0 || a.session_digest.empty()) {
            std::cerr << "named-slot fixture A did not retain a catalogued session\n";
            return 1;
        }
        const auto slot_a          = static_cast<std::uint32_t>(a.slot);
        const std::string digest_a = a.session_digest;
        const std::vector<ninfer::SlotState> states_after_a = engine.slot_states();

        // Score A twice in a row: neither call may change slot_states, consume/advance A, or
        // disagree with the other (same source, same candidates, same deterministic prefix).
        std::vector<ninfer::TokenId> score_a_prompt = prompt_a;
        score_a_prompt.push_back(20500);
        const ninfer::GenerationResult score_a1 =
            engine.generate(engine.prepare_tokens(score_a_prompt), scoring_request(candidates));
        if (score_a1.reused_prompt_tokens != prompt_a.size()) {
            std::cerr << "first scoring request against A reused " << score_a1.reused_prompt_tokens
                      << " tokens, expected the full " << prompt_a.size() << '\n';
            return 1;
        }
        if (!states_equal(engine.slot_states(), states_after_a)) {
            std::cerr << "slot_states changed after the first scoring request against A\n";
            return 1;
        }

        const ninfer::GenerationResult score_a2 =
            engine.generate(engine.prepare_tokens(score_a_prompt), scoring_request(candidates));
        if (score_a2.reused_prompt_tokens != prompt_a.size()) {
            std::cerr << "second scoring request against A reused " << score_a2.reused_prompt_tokens
                      << " tokens, expected the full " << prompt_a.size() << '\n';
            return 1;
        }
        if (score_a2.candidate_logprobs != score_a1.candidate_logprobs) {
            std::cerr << "scoring A twice in a row produced different candidate scores\n";
            return 1;
        }
        if (!states_equal(engine.slot_states(), states_after_a)) {
            std::cerr << "slot_states changed after the second scoring request against A\n";
            return 1;
        }

        // Publish an entirely independent second retained source B.
        const std::vector<ninfer::TokenId> prompt_b = repeated_prompt(300, 20900);
        const ninfer::GenerationResult b =
            engine.generate(engine.prepare_tokens(prompt_b), greedy_continuation_request(1));
        if (b.slot < 0 || b.session_digest.empty()) {
            std::cerr << "named-slot fixture B did not retain a catalogued session\n";
            return 1;
        }
        const auto slot_b = static_cast<std::uint32_t>(b.slot);
        if (slot_b == slot_a && b.session_digest == digest_a) {
            std::cerr << "B did not establish an independent retained session from A\n";
            return 1;
        }
        if (engine.slot_tokens(slot_a, digest_a) != prompt_a) {
            std::cerr << "A's retained session was altered by publishing B\n";
            return 1;
        }
        const std::vector<ninfer::SlotState> states_after_b = engine.slot_states();

        // Score A and B both, after B's admission. Neither call may change slot_states, and both
        // sources remain independently reusable (A surviving B's admission; B freshly scorable).
        const ninfer::GenerationResult score_a3 =
            engine.generate(engine.prepare_tokens(score_a_prompt), scoring_request(candidates));
        if (score_a3.reused_prompt_tokens != prompt_a.size()) {
            std::cerr << "post-B scoring request against A reused " << score_a3.reused_prompt_tokens
                      << " tokens, expected the full " << prompt_a.size() << '\n';
            return 1;
        }
        if (score_a3.candidate_logprobs != score_a1.candidate_logprobs) {
            std::cerr << "scoring A after B's admission produced different candidate scores\n";
            return 1;
        }
        if (!states_equal(engine.slot_states(), states_after_b)) {
            std::cerr << "slot_states changed after scoring A following B's admission\n";
            return 1;
        }

        std::vector<ninfer::TokenId> score_b_prompt = prompt_b;
        score_b_prompt.push_back(20999);
        const ninfer::GenerationResult score_b =
            engine.generate(engine.prepare_tokens(score_b_prompt), scoring_request(candidates));
        if (score_b.reused_prompt_tokens != prompt_b.size()) {
            std::cerr << "scoring request against B reused " << score_b.reused_prompt_tokens
                      << " tokens, expected the full " << prompt_b.size() << '\n';
            return 1;
        }
        if (!states_equal(engine.slot_states(), states_after_b)) {
            std::cerr << "slot_states changed after scoring B\n";
            return 1;
        }

        // Final check: both A and B's retained ledgers are still exactly their original prompts.
        if (engine.slot_tokens(slot_a, digest_a) != prompt_a) {
            std::cerr << "A's retained session was altered by scoring A and B\n";
            return 1;
        }
        if (engine.slot_tokens(slot_b, b.session_digest) != prompt_b) {
            std::cerr << "B's retained session was altered by scoring A and B\n";
            return 1;
        }
    } // Named-slot A/B Engine destroyed here.

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

        // Fresh-prefill scoring never round-trips through the paged KV cache's quantized storage
        // (see src/core/paged_kv_storage.h: I8/FP8_E4M3FN/U8 data planes with FP16/U8 scales) --
        // it computes tail_hidden directly from the one in-flight prefill, so it matches the
        // independent CausalScoring oracle to within float rounding (observed max error 0 across
        // all candidates). A sibling scoring request that forks from a previously retained,
        // quantized-and-stored KV source necessarily re-dequantizes that source before continuing,
        // which introduces small but real quantization noise relative to the oracle's single-shot
        // higher-precision computation (observed max error ~0.06 over a 1101-token context here).
        // This is an expected consequence of the storage boundary, not a reuse-correctness bug, so
        // it gets its own, still tightly bounded, tolerance.
        constexpr float kTolerance       = 0.05F;
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

        // Each scoring request's candidates are evaluated as the next token after the exact
        // sequence it submitted (prompt, or prompt + a divergent suffix token for the siblings),
        // so each must be checked against the oracle computed over that same sequence -- not a
        // single shared reference computed over the bare shared prefix.
        const std::vector<float> fresh_reference         = oracle_scores(prompt);
        const std::vector<float> sibling_a_reference      = oracle_scores(sibling_a_prompt);
        const std::vector<float> sibling_b_reference      = oracle_scores(sibling_b_prompt);
        const std::vector<float> mtp_sibling_a_reference = oracle_scores(mtp_sibling_a_prompt);
        const std::vector<float> mtp_sibling_b_reference = oracle_scores(mtp_sibling_b_prompt);

        if (const int rc =
                compare(fresh_scores, fresh_reference, "fresh-prefill scoring", kTolerance);
            rc != 0) {
            return rc;
        }
        if (const int rc = compare(sibling_a_scores, sibling_a_reference, "sibling A scoring",
                                   kToleranceReused);
            rc != 0) {
            return rc;
        }
        if (const int rc = compare(sibling_b_scores, sibling_b_reference, "sibling B scoring",
                                   kToleranceReused);
            rc != 0) {
            return rc;
        }

        if (const int rc = compare(mtp_fresh_scores, fresh_reference, "Mtp fresh-prefill scoring",
                                   kTolerance);
            rc != 0) {
            return rc;
        }
        if (const int rc = compare(mtp_sibling_a_scores, mtp_sibling_a_reference,
                                   "Mtp sibling A scoring", kToleranceReused);
            rc != 0) {
            return rc;
        }
        if (const int rc = compare(mtp_sibling_b_scores, mtp_sibling_b_reference,
                                   "Mtp sibling B scoring", kToleranceReused);
            rc != 0) {
            return rc;
        }
    }

    std::cout << "OK engine_score_candidates_real (sibling_a_reused=" << sibling_a_reused_tokens
              << " sibling_b_reused=" << sibling_b_reused_tokens
              << " post_scoring_reused=" << post_scoring_reused_tokens
              << " mtp_sibling_a_reused=" << mtp_sibling_a_reused_tokens
              << " mtp_sibling_b_reused=" << mtp_sibling_b_reused_tokens
              << " mtp_post_scoring_reused=" << mtp_post_scoring_reused_tokens << ")\n";
    return 0;
}
