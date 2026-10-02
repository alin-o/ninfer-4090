// Schema/behavior tests for the SystemOne routes (/v1/systemone, /v1/systemone/batch): request
// validation and status codes, prompt/label token-array planning (with a recording fake tokenizer,
// since no live tokenizer/model is available to this test), and the answer/confidence math. No
// Engine/GPU dependency.

#include "serve/request.h"
#include "serve/systemone_answer.h"
#include "serve/systemone_prompt.h"
#include "serve/systemone_request.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

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

Json noul_question(const char* instructions = "is it raining?") {
    return Json{{"type", "noul"}, {"instructions", instructions}};
}

Json choice_question(int options) {
    Json criteria = Json::object();
    for (int i = 0; i < options; ++i) { criteria[std::string(1, char('a' + i))] = Json(nullptr); }
    return Json{{"type", "choice"}, {"instructions", "pick one"}, {"criteria", criteria}};
}

Json score_question(int levels) {
    Json criteria = Json::array();
    for (int i = 0; i < levels; ++i) { criteria.push_back("level " + std::to_string(i)); }
    return Json{{"type", "score"}, {"instructions", "rate it"}, {"criteria", criteria}};
}

int test_state_serialization_and_unwrap() {
    int failures = 0;
    failures += check(serialize_systemone_state(Json("hello")) == "hello", "string state passes through");
    failures += check(serialize_systemone_state(Json{{"body", "wrapped"}}) == "wrapped",
                      "a lone {body: X} wrapper unwraps to X");
    const std::string dumped = serialize_systemone_state(Json{{"a", 1}, {"b", 2}});
    failures += check(Json::parse(dumped) == (Json{{"a", 1}, {"b", 2}}),
                      "a plain object state round-trips through JSON serialization");
    failures += check(serialize_systemone_state(Json{{"body", "x"}, {"other", "y"}}) !=
                          std::string("x"),
                      "a {body, other} object with an extra key is NOT unwrapped (not a lone body)");
    return failures;
}

int test_request_validation_status_codes() {
    int failures = 0;

    failures += check(api_error([] { (void)parse_systemone_request(Json{{"questions", Json::object()}}); })
                              .status == 400,
                      "missing state is a 400");
    failures += check(
        api_error([] { (void)parse_systemone_request(Json{{"state", "s"}}); }).status == 400,
        "missing questions is a 400");

    const std::string huge_state(kSystemOneMaxStateChars + 1, 'x');
    failures += check(api_error([&] {
                          (void)parse_systemone_request(
                              Json{{"state", huge_state}, {"questions", Json::object()}});
                      }).status == 413,
                      "oversized state is a 413");

    Json too_many_questions = Json::object();
    for (std::size_t i = 0; i < kSystemOneMaxQuestions + 1; ++i) {
        too_many_questions[std::to_string(i)] = noul_question();
    }
    failures += check(api_error([&] {
                          (void)parse_systemone_request(Json{{"state", "s"}, {"questions", too_many_questions}});
                      }).status == 413,
                      "too many questions is a 413");

    failures += check(api_error([] {
                          (void)parse_systemone_request(
                              Json{{"state", "s"},
                                   {"questions", Json{{"q1", Json{{"type", "bogus"},
                                                                  {"instructions", "x"}}}}}});
                      }).status == 422,
                      "an unknown question type is a 422");
    failures += check(api_error([] {
                          (void)parse_systemone_request(
                              Json{{"state", "s"},
                                   {"questions", Json{{"q1", Json{{"type", "choice"},
                                                                  {"instructions", "x"}}}}}});
                      }).status == 422,
                      "a choice question without criteria is a 422");

    Json too_many_options = choice_question(static_cast<int>(kSystemOneMaxChoiceOptions) + 1);
    failures += check(api_error([&] {
                          (void)parse_systemone_request(
                              Json{{"state", "s"}, {"questions", Json{{"q1", too_many_options}}}});
                      }).status == 413,
                      "too many choice options is a 413");

    Json too_many_levels = score_question(static_cast<int>(kSystemOneMaxScoreLevels) + 1);
    failures += check(api_error([&] {
                          (void)parse_systemone_request(
                              Json{{"state", "s"}, {"questions", Json{{"q1", too_many_levels}}}});
                      }).status == 413,
                      "too many score levels is a 413");

    const SystemOneRequest ok = parse_systemone_request(
        Json{{"state", "s"},
             {"questions", Json{{"q1", noul_question()}, {"q2", choice_question(3)}}}});
    failures += check(ok.questions.size() == 2, "two valid questions parsed");
    failures += check(ok.questions[0].type == SystemOneQuestionType::Noul &&
                          ok.questions[0].options.size() == 2,
                      "noul question gets exactly the [false, true] option pair");
    failures += check(ok.questions[1].options.size() == 3, "choice question keeps its 3 options");

    failures += check(api_error([] { (void)parse_systemone_batch_request(Json{{"questions", Json::object()}}); })
                              .status == 400,
                      "batch missing states is a 400");
    Json too_many_states = Json::array();
    for (std::size_t i = 0; i <= kSystemOneMaxBatchStates; ++i) { too_many_states.push_back("s"); }
    failures += check(api_error([&] {
                          (void)parse_systemone_batch_request(
                              Json{{"states", too_many_states}, {"questions", Json{{"q", noul_question()}}}});
                      }).status == 413,
                      "too many batch states is a 413");

    return failures;
}

// Recording fake tokenizer: assigns one fresh token id per distinct literal string it has been
// asked to tokenize, so build_systemone_prompt's concatenation boundaries can be checked exactly
// (no live tokenizer/model is available to this task).
class RecordingTokenizer {
public:
    std::vector<ninfer::TokenId> operator()(std::string_view text) {
        calls_.emplace_back(text);
        const std::string key(text);
        auto it = ids_.find(key);
        if (it != ids_.end()) { return it->second; }
        // One synthetic token per UTF-8 byte that is not a continuation byte, so distinct calls
        // never collide, and single-character label strings resolve to exactly one token --
        // matching what a real single-byte/single-codepoint label token looks like.
        std::vector<ninfer::TokenId> tokens;
        for (const unsigned char byte : key) {
            if ((byte & 0xC0) != 0x80) { tokens.push_back(next_id_++); }
        }
        ids_.emplace(key, tokens);
        return tokens;
    }

    [[nodiscard]] const std::vector<std::string>& calls() const { return calls_; }

private:
    std::vector<std::string> calls_;
    std::unordered_map<std::string, std::vector<ninfer::TokenId>> ids_;
    ninfer::TokenId next_id_ = 1;
};

int test_prompt_boundary_safety() {
    int failures = 0;
    RecordingTokenizer tokenizer;
    SystemOneTokenizeFn tokenize = [&tokenizer](std::string_view text) { return tokenizer(text); };

    // SystemOneLabelTable's per-index label-token caches are process-lifetime (static), by
    // design: there is exactly one resident model, so a label token id never changes once
    // resolved. Resolution itself tokenizes the *bare* label spelling ("A", maybe " A") to find
    // the single output token the model would emit -- that text is never part of the prompt
    // array and is not required to end in a newline. Warm it here through a throwaway tokenizer
    // so those resolution-only calls land in a different RecordingTokenizer's log, not the one
    // below whose calls we check against the prompt-concatenation newline-boundary invariant.
    {
        RecordingTokenizer warmup_tokenizer;
        SystemOneTokenizeFn warmup_tokenize = [&warmup_tokenizer](std::string_view text) {
            return warmup_tokenizer(text);
        };
        SystemOneLabelTable warmup_labels(warmup_tokenize);
        (void)warmup_labels.choice_letter_token(0);
        (void)warmup_labels.choice_letter_token(1);
    }

    SystemOneLabelTable labels(tokenize);

    SystemOneQuestion question;
    question.id           = "q1";
    question.type         = SystemOneQuestionType::Choice;
    question.instructions = "pick the best option";
    question.options      = {SystemOneOption{"alpha", "the first option"},
                             SystemOneOption{"beta", "the second option"}};

    std::vector<ninfer::TokenId> candidate_tokens;
    const std::vector<ninfer::TokenId> empty_checkpoint_ledger;
    const std::vector<ninfer::TokenId> prompt = build_systemone_prompt(
        tokenize, empty_checkpoint_ledger, /*has_state=*/true, "state text here", question,
        SystemOnePromptLayout::StateFirst, labels, candidate_tokens);

    failures += check(!prompt.empty(), "prompt is non-empty");
    failures += check(candidate_tokens.size() == 2, "one candidate token per option");
    failures += check(candidate_tokens[0] != candidate_tokens[1],
                      "distinct options resolve to distinct candidate tokens");

    // The critical cross-call concatenation-safety invariant: every tokenize() call's text, other
    // than possibly the very last one, must end in a newline (or be a verbatim ChatML control
    // string, which always ends in \n here too) -- see systemone_prompt.h's header comment.
    const std::vector<std::string>& calls = tokenizer.calls();
    failures += check(!calls.empty(), "the prompt builder actually calls tokenize()");
    for (std::size_t i = 0; i + 1 < calls.size(); ++i) {
        const bool ends_in_newline = !calls[i].empty() && calls[i].back() == '\n';
        failures +=
            check(ends_in_newline, "tokenize() call " + std::to_string(i) + " ('" + calls[i] +
                                       "') ends in a newline before the next call");
    }

    // Concatenation must be exactly the sum of the individual calls' tokens (no re-tokenization
    // of merged text), i.e. the prompt is built purely by array concatenation. Snapshot the call
    // list first: replaying each call through `tokenizer` below appends to the very vector that
    // `calls` references, so iterating `calls` directly while replaying would invalidate it.
    const std::vector<std::string> calls_snapshot = calls;
    std::size_t total_recorded_tokens              = 0;
    for (const std::string& call : calls_snapshot) { total_recorded_tokens += tokenizer(call).size(); }
    // The ChatML turn terminator ("<|im_end|>\n") is fixed scaffolding used twice per build
    // (once closing the system turn, once closing the user turn). Its second use is a cache hit
    // inside cached_fixed_tokens and therefore does not add a second entry to `calls`, even
    // though it does add a second copy of its tokens to `prompt`; account for that known
    // duplicate here rather than asserting a stronger (and false) one-call-per-occurrence
    // invariant.
    total_recorded_tokens += tokenizer(std::string("<|im_end|>\n")).size();
    failures += check(total_recorded_tokens == prompt.size(),
                      "the final prompt is exactly the concatenation of every tokenize() call");

    // Label scaffolding is last: the final two tokenize() calls before the assistant cue must be
    // the per-option label lines ("A\n" / "B\n"), confirming distinct questions differ only at the
    // tail, as required for prefix-cache sharing.
    bool saw_option_label_line = false;
    for (const std::string& call : calls) {
        if (call == "A\n" || call == "B\n") { saw_option_label_line = true; }
    }
    failures += check(saw_option_label_line, "choice label scaffolding (A\\n/B\\n) is tokenized");

    // Fixed scaffolding is cached: asking for a second prompt must not re-record new calls for the
    // shared ChatML/system/label text, only for the new state/instructions/description text.
    const std::size_t calls_before = calls.size();
    std::vector<ninfer::TokenId> second_candidate_tokens;
    SystemOneQuestion second_question = question;
    second_question.id                = "q2";
    (void)build_systemone_prompt(tokenize, empty_checkpoint_ledger, /*has_state=*/true,
                           "a different state", second_question, SystemOnePromptLayout::StateFirst,
                           labels, second_candidate_tokens);
    // "A\n"/"B\n" are identical fixed scaffolding and must hit the cache (no growth from them);
    // the new state/instructions/description text does still get its own call.
    std::size_t repeated_label_calls = 0;
    for (std::size_t i = calls_before; i < tokenizer.calls().size(); ++i) {
        if (tokenizer.calls()[i] == "A\n" || tokenizer.calls()[i] == "B\n") { ++repeated_label_calls; }
    }
    failures += check(repeated_label_calls == 0,
                      "cached fixed label scaffolding is not re-tokenized on a second prompt");

    return failures;
}

int test_systemone_checkpoint_field() {
    int failures = 0;

    // /v1/systemone: state is optional exactly when checkpoint is present.
    const SystemOneRequest checkpoint_no_state = parse_systemone_request(
        Json{{"checkpoint", Json{{"slot", 2}, {"session_digest", "deadbeef"}}},
             {"questions", Json{{"q1", noul_question()}}}});
    failures += check(checkpoint_no_state.checkpoint.present, "checkpoint object parsed as present");
    failures += check(checkpoint_no_state.checkpoint.slot == 2U, "checkpoint.slot preserved");
    failures += check(checkpoint_no_state.checkpoint.session_digest == "deadbeef",
                      "checkpoint.session_digest preserved");
    failures += check(!checkpoint_no_state.has_state,
                      "state is optional when checkpoint is given, and absent here");

    const SystemOneRequest checkpoint_with_state = parse_systemone_request(
        Json{{"checkpoint", Json{{"slot", 0}, {"session_digest", "x"}}},
             {"state", "also given"},
             {"questions", Json{{"q1", noul_question()}}}});
    failures += check(checkpoint_with_state.has_state && checkpoint_with_state.state == "also given",
                      "state is still honored when given alongside a checkpoint");

    failures += check(api_error([] {
                          (void)parse_systemone_request(
                              Json{{"checkpoint", "not-an-object"},
                                   {"questions", Json{{"q1", noul_question()}}}});
                      }).status == 400,
                      "systemone non-object checkpoint is a 400");
    failures += check(api_error([] {
                          (void)parse_systemone_request(
                              Json{{"checkpoint", Json{{"slot", -1}, {"session_digest", "x"}}},
                                   {"questions", Json{{"q1", noul_question()}}}});
                      }).status == 400,
                      "systemone negative checkpoint.slot is a 400");
    failures += check(api_error([] {
                          (void)parse_systemone_request(
                              Json{{"checkpoint", Json{{"slot", 0}, {"session_digest", ""}}},
                                   {"questions", Json{{"q1", noul_question()}}}});
                      }).status == 400,
                      "systemone empty checkpoint.session_digest is a 400");

    // /v1/systemone/batch: 'states' stays required even with a checkpoint given.
    failures += check(api_error([] {
                          (void)parse_systemone_batch_request(
                              Json{{"checkpoint", Json{{"slot", 0}, {"session_digest", "x"}}},
                                   {"questions", Json{{"q1", noul_question()}}}});
                      }).status == 400,
                      "batch 'states' is required even when a checkpoint is given");

    const SystemOneBatchRequest batch_with_checkpoint = parse_systemone_batch_request(
        Json{{"checkpoint", Json{{"slot", 1}, {"session_digest", "y"}}},
             {"states", Json::array({"s1", "s2"})},
             {"questions", Json{{"q1", noul_question()}}}});
    failures += check(batch_with_checkpoint.checkpoint.present &&
                          batch_with_checkpoint.checkpoint.slot == 1U &&
                          batch_with_checkpoint.checkpoint.session_digest == "y",
                      "batch checkpoint parsed alongside required states");
    failures += check(batch_with_checkpoint.states.size() == 2, "batch states still parsed normally");

    return failures;
}

// A tokenizer that resolves the exact literal "<|im_end|>" to a single reserved control-token id
// (as a real tokenizer must, for the ChatML ledger-closing rule to apply at all) while delegating
// every other text to a RecordingTokenizer, so prompt segment *lengths* stay deterministic and
// comparable across build_systemone_prompt calls regardless of global fixed-text cache state.
class ControlAwareTokenizer {
public:
    static constexpr ninfer::TokenId kImEndTokenId = 9000;

    std::vector<ninfer::TokenId> operator()(std::string_view text) {
        if (text == "<|im_end|>") { return {kImEndTokenId}; }
        return inner_(text);
    }

private:
    RecordingTokenizer inner_;
};

int test_checkpoint_ledger_closing_rule() {
    int failures = 0;
    ControlAwareTokenizer control_tokenizer;
    SystemOneTokenizeFn tokenize = [&](std::string_view text) { return control_tokenizer(text); };
    SystemOneLabelTable labels(tokenize);

    // Choice (not Noul): its labels are single letters ("A", "B", ...), which the byte-counting
    // RecordingTokenizer resolves to exactly one token. Noul's default "Yes"/"No" labels are
    // multi-byte and would not resolve under this fake tokenizer's one-token-per-byte scheme.
    SystemOneQuestion question;
    question.id           = "q1";
    question.type         = SystemOneQuestionType::Choice;
    question.instructions = "is it true?";
    question.options      = {SystemOneOption{"alpha", ""}, SystemOneOption{"beta", ""}};

    const std::vector<ninfer::TokenId> ledger_ends_im_end = {101, 102, 103,
                                                             ControlAwareTokenizer::kImEndTokenId};
    const std::vector<ninfer::TokenId> ledger_not_ending = {101, 102, 103, 777};

    std::vector<ninfer::TokenId> candidates_a;
    const std::vector<ninfer::TokenId> prompt_ends_im_end =
        build_systemone_prompt(tokenize, ledger_ends_im_end, /*has_state=*/true, "state", question,
                               SystemOnePromptLayout::StateFirst, labels, candidates_a);
    std::vector<ninfer::TokenId> candidates_b;
    const std::vector<ninfer::TokenId> prompt_not_ending =
        build_systemone_prompt(tokenize, ledger_not_ending, /*has_state=*/true, "state", question,
                               SystemOnePromptLayout::StateFirst, labels, candidates_b);

    failures += check(prompt_ends_im_end.size() >= ledger_ends_im_end.size() &&
                          std::equal(ledger_ends_im_end.begin(), ledger_ends_im_end.end(),
                                     prompt_ends_im_end.begin()),
                      "checkpoint prompt opens with the resident ledger verbatim");

    // "<|im_end|>\n" (11 ASCII bytes under the recording tokenizer's one-token-per-byte scheme)
    // closes the turn when the ledger does NOT already end in <|im_end|>; a ledger that already
    // ends cleanly only needs the bare "\n" (1 token) to reopen the ChatML stream. Everything else
    // in both prompts is identical (same ledger length, same state/question), so the only size
    // difference between the two prompts must be exactly that 11 - 1 = 10 token gap.
    failures += check(prompt_not_ending.size() == prompt_ends_im_end.size() + 10,
                      "ledger not ending in <|im_end|> is closed with the full turn terminator, "
                      "costing exactly the extra control-token length over a bare newline close");

    return failures;
}

int test_checkpoint_shared_prefix_frontier() {
    int failures = 0;
    ControlAwareTokenizer control_tokenizer;
    SystemOneTokenizeFn tokenize = [&](std::string_view text) { return control_tokenizer(text); };
    SystemOneLabelTable labels(tokenize);

    // Choice, for the same single-token-label reason as test_checkpoint_ledger_closing_rule above.
    SystemOneQuestion question;
    question.id           = "q1";
    question.type         = SystemOneQuestionType::Choice;
    question.instructions = "placeholder";
    question.options      = {SystemOneOption{"alpha", ""}, SystemOneOption{"beta", ""}};

    const auto shares_prefix_up_to = [](const std::vector<ninfer::TokenId>& a,
                                        const std::vector<ninfer::TokenId>& b,
                                        std::uint32_t frontier) {
        return frontier > 0 && frontier <= a.size() && frontier <= b.size() &&
               std::equal(a.begin(), a.begin() + frontier, b.begin());
    };

    const std::vector<ninfer::TokenId> no_ledger;

    // StateFirst, no checkpoint: two questions sharing one state must report the same frontier and
    // share exactly the prefix up to it ([system][state]).
    {
        SystemOneQuestion q1 = question, q2 = question;
        q1.instructions = "question one?";
        q2.instructions = "a very different, much longer second question?";
        std::vector<ninfer::TokenId> c1, c2;
        std::uint32_t f1 = 0, f2 = 0;
        const auto p1 = build_systemone_prompt(tokenize, no_ledger, true, "shared state", q1,
                                               SystemOnePromptLayout::StateFirst, labels, c1, &f1);
        const auto p2 = build_systemone_prompt(tokenize, no_ledger, true, "shared state", q2,
                                               SystemOnePromptLayout::StateFirst, labels, c2, &f2);
        failures += check(f1 == f2, "StateFirst frontier matches across siblings sharing [system][state]");
        failures += check(shares_prefix_up_to(p1, p2, f1),
                          "StateFirst siblings share exactly the prefix up to the frontier");
        failures += check(f1 < p1.size(), "StateFirst frontier leaves a diverging suffix");
    }

    // QuestionFirst, no checkpoint: one question over two states must share [system][question].
    {
        std::vector<ninfer::TokenId> c1, c2;
        std::uint32_t f1 = 0, f2 = 0;
        const auto p1 = build_systemone_prompt(tokenize, no_ledger, true, "state one", question,
                                               SystemOnePromptLayout::QuestionFirst, labels, c1, &f1);
        const auto p2 =
            build_systemone_prompt(tokenize, no_ledger, true, "a totally different state text",
                                   question, SystemOnePromptLayout::QuestionFirst, labels, c2, &f2);
        failures +=
            check(f1 == f2, "QuestionFirst frontier matches across siblings sharing [system][question]");
        failures += check(shares_prefix_up_to(p1, p2, f1),
                          "QuestionFirst siblings share exactly the prefix up to the frontier");
        failures += check(f1 < p1.size(), "QuestionFirst frontier leaves a diverging suffix");
    }

    // has_state=false (checkpoint-only question): the whole checkpoint preamble is the shared
    // prefix; two different questions against the same checkpoint must still agree on the frontier.
    {
        const std::vector<ninfer::TokenId> ledger = {201, 202, ControlAwareTokenizer::kImEndTokenId};
        SystemOneQuestion q1 = question, q2 = question;
        q1.instructions = "about the checkpoint, Q1?";
        q2.instructions = "about the checkpoint, a different and longer Q2?";
        std::vector<ninfer::TokenId> c1, c2;
        std::uint32_t f1 = 0, f2 = 0;
        const auto p1 = build_systemone_prompt(tokenize, ledger, /*has_state=*/false, "", q1,
                                               SystemOnePromptLayout::StateFirst, labels, c1, &f1);
        const auto p2 = build_systemone_prompt(tokenize, ledger, /*has_state=*/false, "", q2,
                                               SystemOnePromptLayout::StateFirst, labels, c2, &f2);
        failures += check(f1 == f2,
                          "checkpoint-only (has_state=false) frontier matches across siblings");
        failures += check(shares_prefix_up_to(p1, p2, f1),
                          "checkpoint-only siblings share exactly the checkpoint-preamble prefix");
        failures += check(f1 < p1.size(), "checkpoint-only frontier leaves a diverging suffix");
    }

    return failures;
}

int test_answer_mapping() {
    int failures = 0;

    SystemOneQuestion noul;
    noul.type    = SystemOneQuestionType::Noul;
    noul.options = {SystemOneOption{"false", ""}, SystemOneOption{"true", ""}};
    // candidate_logprobs is float (matching GenerationResult), so tolerances below account for
    // float round-trip precision (~1e-7), not just the probability/confidence math.
    const Json noul_answer = make_systemone_answer(noul, {std::log(0.2f), std::log(0.8f)});
    failures += check(noul_answer["type"] == "noul", "noul answer type");
    failures += check(std::abs(double(noul_answer["noul"]) - 0.8) < 1e-6, "noul = P(Yes)");
    failures += check(std::abs(double(noul_answer["confidence"]) - 0.6) < 1e-6,
                      "noul confidence = clamp(2*peak-1, 0, 1) for peak 0.8");

    SystemOneQuestion choice;
    choice.type    = SystemOneQuestionType::Choice;
    choice.options = {SystemOneOption{"red", ""}, SystemOneOption{"green", ""},
                      SystemOneOption{"blue", ""}};
    const Json choice_answer =
        make_systemone_answer(choice, {std::log(0.1f), std::log(0.7f), std::log(0.2f)});
    failures += check(choice_answer["type"] == "choice", "choice answer type");
    failures += check(choice_answer["choice"] == "green", "choice picks the argmax option name");
    failures += check(choice_answer["probabilities"].contains("red") &&
                          choice_answer["probabilities"].contains("green") &&
                          choice_answer["probabilities"].contains("blue"),
                      "choice probabilities are keyed by option name");
    const double n4 = 3.0, peak4 = 0.7;
    failures += check(std::abs(double(choice_answer["confidence"]) - (n4 * peak4 - 1.0) / (n4 - 1.0)) <
                          1e-6,
                      "choice confidence matches confidence_from_peak(peak, option_count)");

    SystemOneQuestion score;
    score.type    = SystemOneQuestionType::Score;
    score.options = {SystemOneOption{"0", "low"}, SystemOneOption{"1", "mid"},
                     SystemOneOption{"2", "high"}};
    const Json score_answer =
        make_systemone_answer(score, {std::log(0.2f), std::log(0.3f), std::log(0.5f)});
    failures += check(score_answer["type"] == "score", "score answer type");
    failures += check(std::abs(double(score_answer["score"]) - (0 * 0.2 + 1 * 0.3 + 2 * 0.5)) < 1e-6,
                      "score = sum i*p_i");
    failures += check(score_answer["legend"]["1"] == "mid", "score legend keyed by level index");
    failures += check(score_answer["probabilities"].contains("2"),
                      "score probabilities keyed by level index");

    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_state_serialization_and_unwrap();
    failures += test_request_validation_status_codes();
    failures += test_prompt_boundary_safety();
    failures += test_systemone_checkpoint_field();
    failures += test_checkpoint_ledger_closing_rule();
    failures += test_checkpoint_shared_prefix_frontier();
    failures += test_answer_mapping();
    if (failures == 0) { std::cout << "SystemOne tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
