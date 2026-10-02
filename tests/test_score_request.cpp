// Schema/behavior tests for /v1/score request parsing, response rendering, and the restricted
// softmax math shared with SystemOne. No Engine/GPU dependency.

#include "serve/request.h"
#include "serve/restricted_softmax.h"
#include "serve/score_request.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <string>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        (void)function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

int test_parse_prompt_variants() {
    int failures = 0;

    const ScoreRequest string_prompt =
        parse_score_request(Json{{"prompt", "hello"}, {"candidates", Json::array({"a", "b"})}});
    failures += check(string_prompt.prompt_is_tokens == false, "string prompt is not tokens");
    failures += check(string_prompt.prompt_text == "hello", "string prompt text preserved");
    failures += check(string_prompt.candidates.size() == 2, "two string candidates parsed");
    failures +=
        check(string_prompt.candidates[0].is_token_id == false, "string candidate is not a token id");

    const ScoreRequest token_prompt =
        parse_score_request(Json{{"prompt", Json::array({1, 2, 3})}, {"candidates", Json::array({42})}});
    failures += check(token_prompt.prompt_is_tokens == true, "array prompt is tokens");
    failures += check(token_prompt.prompt_token_ids == std::vector<ninfer::TokenId>({1, 2, 3}),
                      "array prompt token ids preserved in order");
    failures += check(token_prompt.candidates[0].is_token_id == true, "int candidate is a token id");
    failures += check(token_prompt.candidates[0].token_id == 42, "int candidate token id preserved");

    failures += check(api_error([] { (void)parse_score_request(Json{{"candidates", Json::array({"a"})}}); })
                              .status == 400,
                      "missing prompt is a 400");
    failures += check(api_error([] { (void)parse_score_request(Json{{"prompt", "x"}}); }).status == 400,
                      "missing candidates is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(Json{{"prompt", "x"}, {"candidates", Json::array()}});
                      }).status == 400,
                      "empty candidates is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(
                              Json{{"prompt", Json::array()}, {"candidates", Json::array({"a"})}});
                      }).status == 400,
                      "empty token prompt is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(
                              Json{{"prompt", "x"}, {"candidates", Json::array({-1})}});
                      }).status == 400,
                      "negative candidate token id is a 400");

    Json too_many = Json::array();
    for (std::size_t i = 0; i < kScoreMaxCandidates + 1; ++i) { too_many.push_back(static_cast<int>(i)); }
    failures += check(
        api_error([&] { (void)parse_score_request(Json{{"prompt", "x"}, {"candidates", too_many}}); }).status ==
            400,
        "over-limit candidate count is a 400");

    return failures;
}

int test_response_rendering() {
    int failures = 0;
    ScoreResponseCandidate a;
    a.has_token = true;
    a.token     = "yes";
    a.token_id  = 10;
    a.logprob   = -0.1;
    ScoreResponseCandidate b;
    b.token_id = 11;
    b.logprob  = -2.0;
    const Json rendered =
        Json::parse(make_score_response("model-x", 7, 3, {a, b}, 0.9));
    failures += check(rendered["object"] == "score", "object field is score");
    failures += check(rendered["model"] == "model-x", "model echoed");
    failures += check(rendered["prompt_tokens"] == 7, "prompt_tokens echoed");
    failures += check(rendered["cached_tokens"] == 3, "cached_tokens echoed");
    failures += check(rendered["candidates"][0]["token"] == "yes", "string candidate token rendered");
    failures += check(!rendered["candidates"][1].contains("token"),
                      "int-only candidate has no token field (no detokenize route)");
    failures += check(std::abs(double(rendered["candidate_mass"]) - 0.9) < 1e-12,
                      "candidate_mass echoed");
    return failures;
}

int test_restricted_softmax_math() {
    int failures = 0;
    // logprobs is intentionally float (matching GenerationResult::candidate_logprobs), so the
    // tolerance below accounts for float round-trip precision (~1e-7), not just the softmax math.
    const std::vector<float> logprobs{std::log(0.2), std::log(0.3)};
    const std::vector<double> probabilities = restricted_softmax(logprobs);
    failures += check(std::abs(probabilities[0] - 0.4) < 1e-6 &&
                          std::abs(probabilities[1] - 0.6) < 1e-6,
                      "restricted softmax renormalizes two candidates to sum to one");
    failures += check(std::abs(candidate_probability_mass(logprobs) - 0.5) < 1e-6,
                      "candidate_probability_mass is the raw (unnormalized) sum");
    failures += check(argmax_index(probabilities) == 1, "argmax picks the larger probability");

    failures += check(std::abs(confidence_from_peak(1.0, 2) - 1.0) < 1e-12,
                      "a certain peak has confidence 1");
    failures += check(std::abs(confidence_from_peak(0.5, 2) - 0.0) < 1e-12,
                      "a uniform two-way peak has confidence 0");
    failures += check(std::abs(confidence_from_peak(1.0, 1) - 1.0) < 1e-12,
                      "a single-candidate question is always fully confident");
    const double n   = 4.0;
    const double peak = 0.7;
    failures += check(std::abs(confidence_from_peak(peak, 4) - (n * peak - 1.0) / (n - 1.0)) < 1e-12,
                      "confidence_from_peak matches clamp((n*peak-1)/(n-1), 0, 1)");
    return failures;
}

int test_parse_checkpoint_field() {
    int failures = 0;

    const ScoreRequest no_checkpoint =
        parse_score_request(Json{{"prompt", "hi"}, {"candidates", Json::array({"a"})}});
    failures += check(!no_checkpoint.checkpoint.present,
                      "absent checkpoint yields CheckpointRef{} (not present)");

    const ScoreRequest null_checkpoint = parse_score_request(
        Json{{"prompt", "hi"}, {"candidates", Json::array({"a"})}, {"checkpoint", nullptr}});
    failures += check(!null_checkpoint.checkpoint.present, "null checkpoint is not present");

    const ScoreRequest with_checkpoint = parse_score_request(
        Json{{"prompt", "hi"},
             {"candidates", Json::array({"a"})},
             {"checkpoint", Json{{"slot", 3}, {"session_digest", "abc123"}}}});
    failures += check(with_checkpoint.checkpoint.present, "valid checkpoint object is present");
    failures += check(with_checkpoint.checkpoint.slot == 3U, "checkpoint.slot preserved");
    failures += check(with_checkpoint.checkpoint.session_digest == "abc123",
                      "checkpoint.session_digest preserved");

    failures += check(api_error([] {
                          (void)parse_score_request(Json{{"prompt", "hi"},
                                                         {"candidates", Json::array({"a"})},
                                                         {"checkpoint", "not-an-object"}});
                      }).status == 400,
                      "non-object checkpoint is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(
                              Json{{"prompt", "hi"},
                                   {"candidates", Json::array({"a"})},
                                   {"checkpoint", Json{{"session_digest", "abc"}}}});
                      }).status == 400,
                      "checkpoint missing slot is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(
                              Json{{"prompt", "hi"},
                                   {"candidates", Json::array({"a"})},
                                   {"checkpoint",
                                    Json{{"slot", -1}, {"session_digest", "abc"}}}});
                      }).status == 400,
                      "negative checkpoint.slot is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(
                              Json{{"prompt", "hi"},
                                   {"candidates", Json::array({"a"})},
                                   {"checkpoint", Json{{"slot", "x"}, {"session_digest", "abc"}}}});
                      }).status == 400,
                      "non-integer checkpoint.slot is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(Json{{"prompt", "hi"},
                                                         {"candidates", Json::array({"a"})},
                                                         {"checkpoint", Json{{"slot", 1}}}});
                      }).status == 400,
                      "checkpoint missing session_digest is a 400");
    failures += check(api_error([] {
                          (void)parse_score_request(
                              Json{{"prompt", "hi"},
                                   {"candidates", Json::array({"a"})},
                                   {"checkpoint", Json{{"slot", 1}, {"session_digest", ""}}}});
                      }).status == 400,
                      "empty checkpoint.session_digest is a 400");

    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_prompt_variants();
    failures += test_response_rendering();
    failures += test_restricted_softmax_math();
    failures += test_parse_checkpoint_field();
    if (failures == 0) { std::cout << "score request/response tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
