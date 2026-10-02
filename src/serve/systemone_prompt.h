#pragma once

// Token-array prompt construction for SystemOne. Performance-critical: fixed ChatML scaffolding
// (system header, assistant suffix, per-label lines) is tokenized once and cached for the life of
// the process; only state/instructions/option-description text is tokenized per request.
//
// Tokenization-boundary rule (see AGENTS-level task brief): concatenating the token arrays from
// two separate tokenize() calls is only safe when the first call's text ends in a newline (or is
// itself an exact special-token string) -- an arbitrary BPE tokenizer may encode a token
// differently depending on what follows it, but a newline is reliably its own token boundary for
// this artifact's tokenizer family, and ChatML's `<|im_start|>`/`<|im_end|>` control strings are
// always single exact tokens. Every helper below is built so each individual tokenize() call's
// input text ends with "\n", except (harmlessly) the final call of the whole prompt. This is
// verified by tests/serve/test_systemone_prompt.cpp using a recording fake tokenizer, since no
// live tokenizer is available to this task.

#include "serve/systemone.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

using SystemOneTokenizeFn = std::function<std::vector<ninfer::TokenId>(std::string_view)>;

enum class SystemOnePromptLayout : std::uint8_t {
    // [system][state][question] -- shares [system][state] across questions on one state.
    StateFirst,
    // [system][question][state] -- shares [system][question] across states for one question.
    QuestionFirst,
};

// One resolved prompt label: the wire-facing option name (SystemOneOption::name) together with
// the exact single-token id the model must emit to select it, and the literal text that token was
// resolved from (for diagnostics / error messages).
struct SystemOneLabelToken {
    std::string option_name;
    std::string label_text;
    ninfer::TokenId token_id = 0;
};

// Resolves and caches (process-lifetime) the single-token ids for every prompt label this
// deployment can ever need: A..Z, 0..9, and the noul Yes/No pair. Resolution tries a small set of
// textual spellings per label (with/without a leading space, as a ChatML assistant turn may or may
// not insert one before the first content token) and keeps the first spelling that tokenizes to
// exactly one token. Throws systemone_error(500, ...) if a label the request actually needs has no
// single-token spelling -- this is a property of the artifact's tokenizer and cannot be worked
// around generically.
class SystemOneLabelTable {
public:
    explicit SystemOneLabelTable(const SystemOneTokenizeFn& tokenize) : tokenize_(tokenize) {}

    [[nodiscard]] ninfer::TokenId choice_letter_token(std::size_t index); // 0 -> 'A', 1 -> 'B', ...
    [[nodiscard]] ninfer::TokenId score_digit_token(std::size_t index);   // 0 -> '0', 1 -> '1', ...
    [[nodiscard]] ninfer::TokenId noul_token(bool value);                 // false -> "No", true -> "Yes"

    [[nodiscard]] char choice_letter_text(std::size_t index) const;
    [[nodiscard]] char score_digit_text(std::size_t index) const;
    [[nodiscard]] std::string_view noul_text(bool value) const;

private:
    [[nodiscard]] ninfer::TokenId resolve_single_token(const std::string& canonical_text);

    const SystemOneTokenizeFn& tokenize_;
};

// Builds the token-array prompt for one (state, question) pair. `state` and
// `question.instructions`/option descriptions are tokenized fresh (they are per-request, variable
// text); everything else comes from a process-wide fixed-segment cache.
//
// `checkpoint_ledger` is empty for a normal (non-checkpoint) prompt, which gets the usual
// [system][state|question][turn_end][assistant_cue] layout. When non-empty, it is
// GenerationService::slot_tokens' resident-session ledger for a named checkpoint: the prompt
// instead opens with that ledger, closes its last (possibly incomplete) turn -- "\n" if the
// ledger's last token is already <|im_end|>, else the full "<|im_end|>\n" -- and continues with
// one new user turn, skipping the system preamble entirely.
//
// `has_state` is false only for a checkpoint prompt whose request omitted "state" (the question is
// about the checkpointed conversation itself); `state` is unused in that case.
//
// `shared_prefix_frontier_out`, when non-null, receives the token offset of the boundary between
// the segment shared by every sibling prompt in the caller's group and the segment that diverges
// per sibling: after [system-or-checkpoint][state] for SystemOnePromptLayout::StateFirst (several
// questions sharing one state) or after [system-or-checkpoint] alone when `has_state` is false
// (several questions sharing one checkpoint with no state), and after
// [system-or-checkpoint][question] for SystemOnePromptLayout::QuestionFirst (one question shared
// across several states, with or without a checkpoint). Callers use this to declare an
// Engine::prepare_tokens shared-prefix frontier on the first prompt of such a group.
[[nodiscard]] std::vector<ninfer::TokenId>
build_systemone_prompt(const SystemOneTokenizeFn& tokenize,
                       const std::vector<ninfer::TokenId>& checkpoint_ledger, bool has_state,
                       std::string_view state, const SystemOneQuestion& question,
                       SystemOnePromptLayout layout, SystemOneLabelTable& labels,
                       std::vector<ninfer::TokenId>& candidate_tokens_out,
                       std::uint32_t* shared_prefix_frontier_out = nullptr);

} // namespace ninfer::serve
