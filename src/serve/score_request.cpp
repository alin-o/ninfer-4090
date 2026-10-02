#include "serve/score_request.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

namespace {

constexpr const char* kPromptParam     = "prompt";
constexpr const char* kCandidatesParam = "candidates";
constexpr const char* kCheckpointParam = "checkpoint";

[[nodiscard]] bool fits_token_id(const RequestJson& value) noexcept {
    if (!value.is_number_integer() && !value.is_number_unsigned()) { return false; }
    const auto signed_value = value.get<std::int64_t>();
    return signed_value >= 0;
}

// Absent/null "checkpoint" yields CheckpointRef{} (not present).
[[nodiscard]] CheckpointRef parse_checkpoint(const RequestJson& body) {
    CheckpointRef checkpoint;
    if (!body.contains(kCheckpointParam) || body.at(kCheckpointParam).is_null()) {
        return checkpoint;
    }
    const RequestJson& value = body.at(kCheckpointParam);
    if (!value.is_object()) {
        bad_request("checkpoint must be an object with 'slot' and 'session_digest'",
                    kCheckpointParam);
    }
    if (!value.contains("slot") || !fits_token_id(value.at("slot"))) {
        bad_request("checkpoint.slot must be a non-negative integer", kCheckpointParam);
    }
    if (!value.contains("session_digest") || !value.at("session_digest").is_string() ||
        value.at("session_digest").get<std::string>().empty()) {
        bad_request("checkpoint.session_digest must be a non-empty string", kCheckpointParam);
    }
    checkpoint.present        = true;
    checkpoint.slot           = static_cast<std::uint32_t>(value.at("slot").get<std::uint64_t>());
    checkpoint.session_digest = value.at("session_digest").get<std::string>();
    return checkpoint;
}

} // namespace

ScoreRequest parse_score_request(const RequestJson& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }

    ScoreRequest request;
    if (body.contains("model") && !body.at("model").is_null()) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        request.model = body.at("model").get<std::string>();
    }

    request.checkpoint = parse_checkpoint(body);

    if (!body.contains(kPromptParam) || body.at(kPromptParam).is_null()) {
        bad_request("prompt is required", kPromptParam);
    }
    const RequestJson& prompt = body.at(kPromptParam);
    if (prompt.is_string()) {
        request.prompt_is_tokens = false;
        request.prompt_text      = prompt.get<std::string>();
    } else if (prompt.is_array()) {
        request.prompt_is_tokens = true;
        request.prompt_token_ids.reserve(prompt.size());
        for (const RequestJson& item : prompt) {
            if (!fits_token_id(item)) {
                bad_request("prompt token ids must be non-negative integers", kPromptParam);
            }
            request.prompt_token_ids.push_back(
                static_cast<ninfer::TokenId>(item.get<std::uint64_t>()));
        }
        if (request.prompt_token_ids.empty()) {
            bad_request("prompt must contain at least one token", kPromptParam);
        }
    } else {
        bad_request("prompt must be a string or an array of token ids", kPromptParam);
    }

    if (!body.contains(kCandidatesParam) || !body.at(kCandidatesParam).is_array()) {
        bad_request("candidates must be an array", kCandidatesParam);
    }
    const RequestJson& candidates = body.at(kCandidatesParam);
    if (candidates.empty()) {
        bad_request("candidates must contain at least one entry", kCandidatesParam);
    }
    if (candidates.size() > kScoreMaxCandidates) {
        bad_request("candidates must contain at most " + std::to_string(kScoreMaxCandidates) +
                        " entries",
                    kCandidatesParam);
    }
    request.candidates.reserve(candidates.size());
    for (const RequestJson& item : candidates) {
        ScoreCandidateSpec candidate;
        if (item.is_string()) {
            candidate.is_token_id = false;
            candidate.text        = item.get<std::string>();
            if (candidate.text.empty()) {
                bad_request("candidate strings must not be empty", kCandidatesParam);
            }
        } else if (fits_token_id(item)) {
            candidate.is_token_id = true;
            candidate.token_id = static_cast<ninfer::TokenId>(item.get<std::uint64_t>());
        } else {
            bad_request("each candidate must be a string or a non-negative token id",
                        kCandidatesParam);
        }
        request.candidates.push_back(std::move(candidate));
    }
    return request;
}

std::string
make_score_response(const std::string& model, std::uint32_t prompt_tokens,
                    std::uint32_t cached_tokens,
                    const std::vector<ScoreResponseCandidate>& candidates, double candidate_mass) {
    nlohmann::json rendered_candidates = nlohmann::json::array();
    for (const ScoreResponseCandidate& candidate : candidates) {
        nlohmann::json entry{
            {"token_id", candidate.token_id},
            {"logprob", candidate.logprob},
            {"probability", candidate.probability},
            {"restricted_probability", candidate.restricted_probability},
        };
        if (candidate.has_token) { entry["token"] = candidate.token; }
        rendered_candidates.push_back(std::move(entry));
    }
    nlohmann::json body{
        {"object", "score"},
        {"model", model},
        {"prompt_tokens", prompt_tokens},
        {"cached_tokens", cached_tokens},
        {"candidates", std::move(rendered_candidates)},
        {"candidate_mass", candidate_mass},
    };
    return body.dump();
}

} // namespace ninfer::serve
