#include "serve/http_server.h"

#include "serve/checkpoint_lookup.h"
#include "serve/http_transport.h"
#include "serve/systemone_answer.h"
#include "serve/systemone_prompt.h"
#include "serve/systemone_request.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace ninfer::serve {
namespace {

// laya-serve answers "server busy, try again later" with 503; this fork's Overloaded mapping is
// 429 (rate_limit_error) for the OpenAI-shaped routes. SystemOne follows the brief's explicit
// laya-compatible status instead, so an Overloaded/QueueTimeout/Unavailable failure is remapped
// here rather than left at the OpenAI routes' status. Checkpoint-lookup failures (409 slot_busy /
// slot_session_mismatch, 400 invalid checkpoint) already carry their own status and are left as-is.
ApiError to_systemone_busy_if_applicable(ApiError error) {
    if (error.status == 429 || error.status == 503) {
        error.status  = 503;
        error.type    = "server_error";
        error.code    = "busy";
        error.message = "server busy, try again later";
    }
    return error;
}

struct ScoredQuestion {
    const SystemOneQuestion* question = nullptr;
    std::vector<ninfer::TokenId> prompt_tokens;
    std::vector<ninfer::TokenId> candidate_tokens;
};

// Submits one shared-prefix group: the first prompt is submitted and awaited alone so its KV
// becomes the reusable prefix (optionally declaring `first_prompt_frontiers` so the Engine
// registers it as a reusable shared-prefix catalog entry), then the remaining prompts are
// submitted together (hitting that cached prefix) before any of them is awaited, so the Engine can
// batch them in one decode round.
std::vector<ScoreResult> run_shared_prefix_group(GenerationService& service,
                                                 const std::vector<ScoredQuestion>& group,
                                                 std::uint32_t& cached_tokens_total,
                                                 const std::function<bool()>& is_cancelled,
                                                 std::vector<std::uint32_t> first_prompt_frontiers = {}) {
    std::vector<ScoreResult> results;
    results.reserve(group.size());
    if (group.empty()) { return results; }

    results.push_back(service.score(group[0].prompt_tokens, group[0].candidate_tokens, is_cancelled,
                                    std::move(first_prompt_frontiers)));
    cached_tokens_total += results[0].cached_tokens;
    if (group.size() == 1) { return results; }

    std::vector<ScorePrepared> prepared;
    prepared.reserve(group.size() - 1);
    for (std::size_t i = 1; i < group.size(); ++i) {
        prepared.push_back(
            service.prepare_score(group[i].prompt_tokens, group[i].candidate_tokens, is_cancelled));
    }
    for (std::size_t i = 1; i < group.size(); ++i) {
        ScoreResult result = service.run_score(prepared[i - 1], is_cancelled);
        cached_tokens_total += result.cached_tokens;
        results.push_back(std::move(result));
    }
    return results;
}

// Answers every question in `questions` about one (checkpoint-or-not, state-or-not) context, using
// SystemOnePromptLayout::StateFirst so [checkpoint-or-system][state] is the shared prefix. When
// `frontier_enabled` and there is more than one sibling question, the first prompt declares that
// boundary (build_systemone_prompt's `shared_prefix_frontier_out`) so the Engine registers it and
// every later question in the group -- including, for handle_systemone_batch's general multi-state
// path, later STATES reusing the same checkpoint-or-system preamble via the durable/sibling catalog
// -- reuses it without redeclaring.
nlohmann::ordered_json answer_one_state(GenerationService& service,
                                        const std::vector<ninfer::TokenId>& checkpoint_ledger,
                                        bool has_state, const std::string& state,
                                        const std::vector<SystemOneQuestion>& questions,
                                        SystemOnePromptLayout layout, SystemOneLabelTable& labels,
                                        const SystemOneTokenizeFn& tokenize, bool frontier_enabled,
                                        std::uint32_t& input_tokens_total,
                                        std::uint32_t& cached_tokens_total,
                                        const std::function<bool()>& is_cancelled) {
    std::vector<ScoredQuestion> group;
    group.reserve(questions.size());
    std::uint32_t frontier = 0;
    bool have_frontier     = false;
    const bool want_frontier = frontier_enabled && questions.size() > 1;
    for (std::size_t i = 0; i < questions.size(); ++i) {
        const SystemOneQuestion& question = questions[i];
        ScoredQuestion scored;
        scored.question           = &question;
        std::uint32_t* frontier_out = (want_frontier && i == 0) ? &frontier : nullptr;
        scored.prompt_tokens = build_systemone_prompt(tokenize, checkpoint_ledger, has_state, state,
                                                      question, layout, labels,
                                                      scored.candidate_tokens, frontier_out);
        if (frontier_out != nullptr) { have_frontier = true; }
        group.push_back(std::move(scored));
    }

    std::vector<std::uint32_t> first_prompt_frontiers;
    if (have_frontier) { first_prompt_frontiers.push_back(frontier); }

    const std::vector<ScoreResult> results = run_shared_prefix_group(
        service, group, cached_tokens_total, is_cancelled, std::move(first_prompt_frontiers));

    nlohmann::ordered_json answers = nlohmann::ordered_json::object();
    for (std::size_t i = 0; i < group.size(); ++i) {
        input_tokens_total += results[i].prompt_tokens;
        answers[group[i].question->id] =
            make_systemone_answer(*group[i].question, results[i].candidate_logprobs);
    }
    return answers;
}

} // namespace

void HttpServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() > kSystemOneMaxBodyBytes) {
        ApiError error;
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.message = "request body too large";
        write_openai_error(res, error);
        return;
    }

    SystemOneRequest request;
    std::vector<ninfer::TokenId> checkpoint_ledger;
    try {
        request = parse_systemone_request(parse_json_body(req));
        if (request.checkpoint.present) {
            checkpoint_ledger = resolve_checkpoint_ledger(*service_, request.checkpoint);
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, to_systemone_busy_if_applicable(exception.error()));
        return;
    }
    const std::string response_model = !request.model.empty() ? request.model : public_model_id_;

    const SystemOneTokenizeFn tokenize = [this](std::string_view text) {
        return service_->tokenize_text(text);
    };
    SystemOneLabelTable labels(tokenize);
    const auto is_cancelled      = [&req] { return client_disconnected(req); };
    const bool frontier_enabled  = service_->options().score_prefix_cache;

    std::uint32_t input_tokens  = 0;
    std::uint32_t cached_tokens = 0;
    nlohmann::ordered_json answers;
    try {
        // One context (checkpoint and/or state), possibly several questions:
        // [checkpoint-or-system][state-or-nothing] is the shared prefix.
        answers = answer_one_state(*service_, checkpoint_ledger, request.has_state, request.state,
                                   request.questions, SystemOnePromptLayout::StateFirst, labels,
                                   tokenize, frontier_enabled, input_tokens, cached_tokens,
                                   is_cancelled);
    } catch (const ApiException& exception) {
        write_openai_error(res, to_systemone_busy_if_applicable(exception.error()));
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "server_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    nlohmann::ordered_json body;
    body["model"]                     = response_model;
    body["answers"]                   = std::move(answers);
    body["usage"]["input_tokens"]     = input_tokens;
    body["usage"]["output_tokens"]    = 0;
    body["usage"]["cached_tokens"]    = cached_tokens;
    res.set_content(body.dump(), "application/json");
}

void HttpServer::handle_systemone_batch(const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() > kSystemOneMaxBodyBytes) {
        ApiError error;
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.message = "request body too large";
        write_openai_error(res, error);
        return;
    }

    SystemOneBatchRequest request;
    std::vector<ninfer::TokenId> checkpoint_ledger;
    try {
        request = parse_systemone_batch_request(parse_json_body(req));
        if (request.checkpoint.present) {
            checkpoint_ledger = resolve_checkpoint_ledger(*service_, request.checkpoint);
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, to_systemone_busy_if_applicable(exception.error()));
        return;
    }
    const std::string response_model = !request.model.empty() ? request.model : public_model_id_;

    const SystemOneTokenizeFn tokenize = [this](std::string_view text) {
        return service_->tokenize_text(text);
    };
    SystemOneLabelTable labels(tokenize);
    const auto is_cancelled     = [&req] { return client_disconnected(req); };
    const bool frontier_enabled = service_->options().score_prefix_cache;

    nlohmann::ordered_json results        = nlohmann::ordered_json::array();
    std::uint32_t total_input_tokens  = 0;
    std::uint32_t total_cached_tokens = 0;
    try {
        if (request.questions.size() == 1) {
            // One question, several states: [checkpoint-or-system][question] is the shared
            // prefix, shared across every state by swapping the question block before the state
            // block (QuestionFirst layout). `states` is always required, so this group always has
            // at least one member regardless of checkpoint/has_state.
            const SystemOneQuestion& question = request.questions.front();
            std::vector<ScoredQuestion> group;
            group.reserve(request.states.size());
            std::uint32_t frontier        = 0;
            bool have_frontier            = false;
            const bool want_frontier = frontier_enabled && request.states.size() > 1;
            for (std::size_t i = 0; i < request.states.size(); ++i) {
                const std::string& state = request.states[i];
                ScoredQuestion scored;
                scored.question           = &question;
                std::uint32_t* frontier_out = (want_frontier && i == 0) ? &frontier : nullptr;
                scored.prompt_tokens = build_systemone_prompt(
                    tokenize, checkpoint_ledger, /*has_state=*/true, state, question,
                    SystemOnePromptLayout::QuestionFirst, labels, scored.candidate_tokens,
                    frontier_out);
                if (frontier_out != nullptr) { have_frontier = true; }
                group.push_back(std::move(scored));
            }
            std::vector<std::uint32_t> first_prompt_frontiers;
            if (have_frontier) { first_prompt_frontiers.push_back(frontier); }
            const std::vector<ScoreResult> scored_results = run_shared_prefix_group(
                *service_, group, total_cached_tokens, is_cancelled,
                std::move(first_prompt_frontiers));
            for (std::size_t i = 0; i < group.size(); ++i) {
                total_input_tokens += scored_results[i].prompt_tokens;
                nlohmann::ordered_json answers = nlohmann::ordered_json::object();
                answers[question.id] = make_systemone_answer(question, scored_results[i].candidate_logprobs);
                nlohmann::ordered_json entry;
                entry["model"]                  = response_model;
                entry["answers"]                = std::move(answers);
                entry["usage"]["input_tokens"]  = scored_results[i].prompt_tokens;
                entry["usage"]["output_tokens"] = 0;
                entry["usage"]["cached_tokens"] = scored_results[i].cached_tokens;
                results.push_back(std::move(entry));
            }
        } else {
            // General case: state-first layout per state, independent across states. Each state
            // reuses the same checkpoint ledger (read-only); sharing is only within a state's own
            // question group, not across states.
            for (const std::string& state : request.states) {
                std::uint32_t state_input_tokens  = 0;
                std::uint32_t state_cached_tokens = 0;
                nlohmann::ordered_json answers =
                    answer_one_state(*service_, checkpoint_ledger, /*has_state=*/true, state,
                                     request.questions, SystemOnePromptLayout::StateFirst, labels,
                                     tokenize, frontier_enabled, state_input_tokens,
                                     state_cached_tokens, is_cancelled);
                total_input_tokens += state_input_tokens;
                total_cached_tokens += state_cached_tokens;
                nlohmann::ordered_json entry;
                entry["model"]                  = response_model;
                entry["answers"]                = std::move(answers);
                entry["usage"]["input_tokens"]  = state_input_tokens;
                entry["usage"]["output_tokens"] = 0;
                entry["usage"]["cached_tokens"] = state_cached_tokens;
                results.push_back(std::move(entry));
            }
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, to_systemone_busy_if_applicable(exception.error()));
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "server_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    nlohmann::ordered_json body;
    body["model"]                        = response_model;
    body["results"]                      = std::move(results);
    body["total_usage"]["input_tokens"]  = total_input_tokens;
    body["total_usage"]["output_tokens"] = 0;
    body["total_usage"]["cached_tokens"] = total_cached_tokens;
    res.set_content(body.dump(), "application/json");
}

} // namespace ninfer::serve
