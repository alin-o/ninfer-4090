#include "serve/systemone_prompt.h"

#include <array>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninfer::serve {

namespace {

// Process-lifetime cache for scaffolding text that never varies across requests. Guarded by a
// mutex: HTTP handlers run on multiple worker threads, and tokenizing a cache miss must not race.
std::vector<ninfer::TokenId> cached_fixed_tokens(const SystemOneTokenizeFn& tokenize,
                                                 const std::string& text) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::vector<ninfer::TokenId>> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(text);
        if (it != cache.end()) { return it->second; }
    }
    std::vector<ninfer::TokenId> tokens = tokenize(text);
    {
        std::lock_guard<std::mutex> lock(mutex);
        cache.emplace(text, tokens);
    }
    return tokens;
}

void append(std::vector<ninfer::TokenId>& out, const std::vector<ninfer::TokenId>& segment) {
    out.insert(out.end(), segment.begin(), segment.end());
}

// ChatML scaffolding, thinking disabled. Every fixed piece below ends in "\n", so concatenating
// its tokenization against the next call's tokenization never splits a token the live tokenizer
// would otherwise have merged across the boundary (see the header comment).
const std::string& system_header_text() {
    static const std::string text = "<|im_start|>system\n";
    return text;
}
const std::string& turn_end_text() {
    static const std::string text = "<|im_end|>\n";
    return text;
}
const std::string& user_header_text() {
    static const std::string text = "<|im_start|>user\n";
    return text;
}
const std::string& assistant_cue_text() {
    // Thinking disabled: an empty <think> block, exactly like the rest of ninfer-serve's
    // non-reasoning ChatML prompts.
    static const std::string text = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return text;
}
const std::string& system_body_text() {
    static const std::string text =
        "You answer closed-form questions about the record the user gives you. For each "
        "question you are given a short list of labelled options; answer by producing exactly "
        "one token: the label of the single best option, and nothing else -- no punctuation, no "
        "explanation.\n";
    return text;
}

// The resident ledger-closing rule needs to recognize the exact <|im_end|> control token; this is
// cached through the same process-lifetime fixed-text cache as every other control string.
ninfer::TokenId im_end_token_id(const SystemOneTokenizeFn& tokenize) {
    const std::vector<ninfer::TokenId> encoded = cached_fixed_tokens(tokenize, "<|im_end|>");
    if (encoded.size() != 1) {
        throw std::runtime_error(
            "SystemOne checkpoint: \"<|im_end|>\" does not tokenize to a single control token "
            "under this artifact's tokenizer");
    }
    return encoded.front();
}

std::string label_line(std::string_view label, std::string_view description) {
    std::string line(label);
    line.push_back('\n');
    if (!description.empty()) {
        line.append(description);
        line.push_back('\n');
    }
    return line;
}

} // namespace

ninfer::TokenId SystemOneLabelTable::resolve_single_token(const std::string& canonical_text) {
    // A ChatML assistant turn's first content token is sometimes emitted with a leading space by
    // the tokenizer's merge rules and sometimes not; try the bare label first (most common for a
    // single capital letter or digit immediately after a GPT-style BPE tokenizer's newline), then
    // a leading-space spelling. This is the one SystemOne assumption that cannot be checked
    // without a live tokenizer -- resolution fails loudly rather than guessing further.
    const std::array<std::string, 2> spellings{canonical_text, " " + canonical_text};
    for (const std::string& spelling : spellings) {
        const std::vector<ninfer::TokenId> encoded = tokenize_(spelling);
        if (encoded.size() == 1) { return encoded.front(); }
    }
    throw std::runtime_error("SystemOne label '" + canonical_text +
                             "' does not tokenize to a single token under either spelling tried "
                             "(bare or leading-space); this artifact's tokenizer needs an "
                             "explicit label-resolution rule added here");
}

ninfer::TokenId SystemOneLabelTable::choice_letter_token(std::size_t index) {
    static std::mutex mutex;
    static std::unordered_map<std::size_t, ninfer::TokenId> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(index);
        if (it != cache.end()) { return it->second; }
    }
    const ninfer::TokenId token_id = resolve_single_token(std::string(1, choice_letter_text(index)));
    std::lock_guard<std::mutex> lock(mutex);
    cache.emplace(index, token_id);
    return token_id;
}

ninfer::TokenId SystemOneLabelTable::score_digit_token(std::size_t index) {
    static std::mutex mutex;
    static std::unordered_map<std::size_t, ninfer::TokenId> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(index);
        if (it != cache.end()) { return it->second; }
    }
    const ninfer::TokenId token_id = resolve_single_token(std::string(1, score_digit_text(index)));
    std::lock_guard<std::mutex> lock(mutex);
    cache.emplace(index, token_id);
    return token_id;
}

ninfer::TokenId SystemOneLabelTable::noul_token(bool value) {
    static std::mutex mutex;
    static std::unordered_map<bool, ninfer::TokenId> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(value);
        if (it != cache.end()) { return it->second; }
    }
    const ninfer::TokenId token_id = resolve_single_token(std::string(noul_text(value)));
    std::lock_guard<std::mutex> lock(mutex);
    cache.emplace(value, token_id);
    return token_id;
}

char SystemOneLabelTable::choice_letter_text(std::size_t index) const {
    return static_cast<char>('A' + static_cast<int>(index));
}

char SystemOneLabelTable::score_digit_text(std::size_t index) const {
    return static_cast<char>('0' + static_cast<int>(index));
}

std::string_view SystemOneLabelTable::noul_text(bool value) const {
    // NInfer's own default wording (Yes/No); laya's own default is a different pair
    // ("false"/"true") and is not reused here -- see systemone.h's header comment.
    return value ? std::string_view("Yes") : std::string_view("No");
}

std::vector<ninfer::TokenId>
build_systemone_prompt(const SystemOneTokenizeFn& tokenize,
                       const std::vector<ninfer::TokenId>& checkpoint_ledger, bool has_state,
                       std::string_view state, const SystemOneQuestion& question,
                       SystemOnePromptLayout layout, SystemOneLabelTable& labels,
                       std::vector<ninfer::TokenId>& candidate_tokens_out,
                       std::uint32_t* shared_prefix_frontier_out) {
    std::vector<ninfer::TokenId> prompt;

    if (checkpoint_ledger.empty()) {
        append(prompt, cached_fixed_tokens(tokenize, system_header_text()));
        append(prompt, cached_fixed_tokens(tokenize, system_body_text()));
        append(prompt, cached_fixed_tokens(tokenize, turn_end_text()));
        append(prompt, cached_fixed_tokens(tokenize, user_header_text()));
    } else {
        // Named checkpoint: skip the system preamble and continue the retained session with one
        // new user turn. Ledger-closing rule: the ledger's last token may already be <|im_end|>
        // (the checkpointed turn finished cleanly) -- in that case only a newline reopens the
        // ChatML stream; otherwise the previous (possibly incomplete) turn must be force-closed
        // with the full "<|im_end|>\n" before the new user turn can open.
        append(prompt, checkpoint_ledger);
        if (checkpoint_ledger.back() == im_end_token_id(tokenize)) {
            append(prompt, cached_fixed_tokens(tokenize, "\n"));
        } else {
            append(prompt, cached_fixed_tokens(tokenize, turn_end_text()));
        }
        append(prompt, cached_fixed_tokens(tokenize, user_header_text()));
    }

    std::string state_text(state);
    state_text.push_back('\n');

    const auto append_state = [&] { append(prompt, tokenize(state_text)); };
    const auto append_question = [&] {
        std::string instructions = question.instructions;
        instructions.push_back('\n');
        append(prompt, tokenize(instructions));
        append(prompt, cached_fixed_tokens(tokenize, "Options:\n"));
        for (std::size_t i = 0; i < question.options.size(); ++i) {
            std::string label_text;
            ninfer::TokenId candidate_token = 0;
            switch (question.type) {
            case SystemOneQuestionType::Choice:
                label_text      = std::string(1, labels.choice_letter_text(i));
                candidate_token = labels.choice_letter_token(i);
                break;
            case SystemOneQuestionType::Score:
                label_text      = std::string(1, labels.score_digit_text(i));
                candidate_token = labels.score_digit_token(i);
                break;
            case SystemOneQuestionType::Noul:
                label_text      = std::string(labels.noul_text(i == 1));
                candidate_token = labels.noul_token(i == 1);
                break;
            }
            candidate_tokens_out.push_back(candidate_token);
            // Label scaffolding is fixed per-position ("A\n", "B\n", ...) and cached; the
            // option's rubric description, if any, is the per-request variable part.
            append(prompt, cached_fixed_tokens(tokenize, label_text + "\n"));
            if (!question.options[i].description.empty()) {
                std::string description = question.options[i].description;
                description.push_back('\n');
                append(prompt, tokenize(description));
            }
        }
    };

    const auto mark_frontier = [&] {
        if (shared_prefix_frontier_out != nullptr) {
            *shared_prefix_frontier_out = static_cast<std::uint32_t>(prompt.size());
        }
    };

    if (!has_state) {
        // No state segment at all (checkpoint-only question): whatever precedes this point
        // (system-or-checkpoint preamble) is the whole shared prefix across sibling questions.
        mark_frontier();
        append_question();
    } else if (layout == SystemOnePromptLayout::StateFirst) {
        append_state();
        mark_frontier();
        append_question();
    } else {
        append_question();
        mark_frontier();
        append_state();
    }

    append(prompt, cached_fixed_tokens(tokenize, turn_end_text()));
    append(prompt, cached_fixed_tokens(tokenize, assistant_cue_text()));
    return prompt;
}

} // namespace ninfer::serve
