#include "serve/http_server.h"

#include "serve/checkpoint_lookup.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"
#include "serve/restricted_softmax.h"
#include "serve/score_request.h"

#include <cmath>
#include <unordered_set>
#include <utility>

namespace ninfer::serve {
namespace {

// Resolves one wire candidate to a concrete token id. String candidates must tokenize to exactly
// one artifact token (the only shape ExecutionOptions::score_candidates accepts); anything else is
// a 400, not a silent multi-token approximation.
ninfer::TokenId resolve_candidate_token_id(const GenerationService& service,
                                           const ScoreCandidateSpec& candidate) {
    if (candidate.is_token_id) { return candidate.token_id; }
    const std::vector<ninfer::TokenId> encoded = service.tokenize_text(candidate.text);
    if (encoded.size() != 1) {
        bad_request("candidate \"" + candidate.text +
                        "\" does not tokenize to exactly one token",
                    "candidates");
    }
    return encoded.front();
}

} // namespace

void HttpServer::handle_score(const httplib::Request& req, httplib::Response& res) {
    ScoreRequest request;
    std::vector<ninfer::TokenId> prompt_tokens;
    std::vector<ninfer::TokenId> candidate_ids;
    std::vector<ScoreResponseCandidate> response_candidates;
    std::string response_model;
    try {
        request = parse_score_request(parse_json_body(req));
        if (!request.model.empty()) {
            validate_openai_model(request.model, public_model_id_);
            response_model = request.model;
        } else {
            response_model = public_model_id_;
        }

        std::vector<ninfer::TokenId> continuation_tokens =
            request.prompt_is_tokens ? std::move(request.prompt_token_ids)
                                     : service_->tokenize_text(request.prompt_text);
        if (continuation_tokens.empty()) {
            bad_request("prompt tokenized to zero tokens", "prompt");
        }

        if (request.checkpoint.present) {
            prompt_tokens = resolve_checkpoint_ledger(*service_, request.checkpoint);
            prompt_tokens.insert(prompt_tokens.end(), continuation_tokens.begin(),
                                 continuation_tokens.end());
        } else {
            prompt_tokens = std::move(continuation_tokens);
        }

        candidate_ids.reserve(request.candidates.size());
        response_candidates.reserve(request.candidates.size());
        std::unordered_set<ninfer::TokenId> seen_candidates;
        seen_candidates.reserve(request.candidates.size());
        for (const ScoreCandidateSpec& candidate : request.candidates) {
            const ninfer::TokenId token_id = resolve_candidate_token_id(*service_, candidate);
            if (!seen_candidates.insert(token_id).second) {
                bad_request("duplicate candidate token id " + std::to_string(token_id),
                            "candidates");
            }
            candidate_ids.push_back(token_id);

            ScoreResponseCandidate response_candidate;
            response_candidate.token_id = token_id;
            if (!candidate.is_token_id) {
                response_candidate.has_token = true;
                response_candidate.token     = candidate.text;
            }
            response_candidates.push_back(std::move(response_candidate));
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    ScoreResult result;
    try {
        result = service_->score(std::move(prompt_tokens), candidate_ids,
                                 [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    const std::vector<double> restricted = restricted_softmax(result.candidate_logprobs);
    const double candidate_mass          = candidate_probability_mass(result.candidate_logprobs);
    for (std::size_t i = 0; i < response_candidates.size(); ++i) {
        const double logprob                           = double(result.candidate_logprobs[i]);
        response_candidates[i].logprob                 = logprob;
        response_candidates[i].probability              = std::exp(logprob);
        response_candidates[i].restricted_probability    = restricted[i];
    }

    res.set_content(make_score_response(response_model, result.prompt_tokens,
                                        result.cached_tokens, response_candidates, candidate_mass),
                    "application/json");
}

} // namespace ninfer::serve
