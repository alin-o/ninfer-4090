#pragma once

// POST /v1/score: next-token scoring over an explicit candidate set. Wire-format parsing/response
// building for the route; src/serve/score_http.cpp wires it to GenerationService::score.

#include "serve/checkpoint_ref.h"
#include "serve/request_json.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

constexpr std::size_t kScoreMaxCandidates = 1024;

// One candidate exactly as given on the wire: either literal text (must tokenize to exactly one
// token) or an explicit token id.
struct ScoreCandidateSpec {
    bool is_token_id{false};
    std::string text;          // populated when !is_token_id
    ninfer::TokenId token_id{}; // populated when is_token_id
};

struct ScoreRequest {
    std::string model; // optional on the wire; empty means unspecified
    bool prompt_is_tokens{false};
    std::string prompt_text;
    std::vector<ninfer::TokenId> prompt_token_ids;
    std::vector<ScoreCandidateSpec> candidates;
    // Optional named checkpoint: when present, `prompt`/`prompt_token_ids` above is the
    // continuation appended after GenerationService::slot_tokens(checkpoint.slot,
    // checkpoint.session_digest), not the full prompt.
    CheckpointRef checkpoint;
};

// Pure request-shape validation: body structure, prompt string/array-of-ints, candidate count and
// element shapes. Does not tokenize and does not know about the tokenizer's vocabulary, so a
// string candidate's "exactly one token" rule and final candidate-id distinctness are checked
// later, once string candidates are resolved against the artifact tokenizer. Throws ApiException
// (400 invalid_request_error) on malformed input.
[[nodiscard]] ScoreRequest parse_score_request(const RequestJson& body);

struct ScoreResponseCandidate {
    bool has_token{false};
    std::string token; // decoded text, only when the wire candidate was given as a string
    ninfer::TokenId token_id{};
    double logprob{};
    double probability{};            // exp(logprob): true full-vocabulary probability
    double restricted_probability{}; // softmax restricted to just this candidate set
};

// Renders the {"object":"score",...} response body.
[[nodiscard]] std::string
make_score_response(const std::string& model, std::uint32_t prompt_tokens,
                    std::uint32_t cached_tokens,
                    const std::vector<ScoreResponseCandidate>& candidates, double candidate_mass);

} // namespace ninfer::serve
