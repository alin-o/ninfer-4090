#pragma once

// Shared wire-neutral types for the SystemOne question-answering routes (/v1/systemone,
// /v1/systemone/batch). SystemOne asks a small set of closed-form questions (yes/no, single
// choice, or a graded score) about one or more opaque "state" blobs and answers them from
// full-vocabulary next-token log-probabilities (see ScoreResult / GenerationService::score),
// never from generated text.
//
// Protocol note: the original TypeScript reference for this wire format
// (examples/systemone-proxy/systemone.ts) was deleted upstream before it could be read into this
// implementation, with no recoverable git history. This header, and the prompt wording and
// confidence formula implemented alongside it, are an original NInfer design built from the task
// specification text plus the public shape of laya's Python reference (laya/serve.py,
// laya/agent.py, laya/common.py) for protocol *shape* only -- laya's own production backend is a
// non-LLM BERT-style classifier with a masked-token prompt format, so neither its prompt wording
// nor its confidence formulas transfer here; both are reimplemented from scratch for NInfer's
// ChatML/logprob-based model.

#include "serve/request.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

// Throws ApiException(status, "invalid_request_error", message) -- the shared error path for all
// SystemOne request-shape failures. `status` carries the laya-serve-compatible status code for the
// failure class (400 malformed/missing field, 413 over a size limit, 422 invalid question
// definition); the response body uses NInfer's normal error-object rendering
// (write_openai_error), not laya's bare `{"detail": ...}` shape, for consistency with the rest of
// ninfer-serve.
[[noreturn]] void systemone_error(int status, std::string message);

// NInfer-specific deviations from laya-serve's own (looser) limits of 100 choice options / 32
// score levels / 512 total options: NInfer renders each option as a single uppercase letter
// (choice) or single digit (score) label token, so the option count is capped at the size of
// that label alphabet.
inline constexpr std::size_t kSystemOneMaxQuestions     = 64;
inline constexpr std::size_t kSystemOneMaxStateChars    = 50000;
inline constexpr std::size_t kSystemOneMaxBatchStates   = 64;
inline constexpr std::size_t kSystemOneMaxBodyBytes     = 2 * 1024 * 1024;
inline constexpr std::size_t kSystemOneMaxChoiceOptions = 26; // labels A..Z
inline constexpr std::size_t kSystemOneMaxScoreLevels   = 10; // labels 0..9

enum class SystemOneQuestionType : std::uint8_t {
    Noul,
    Choice,
    Score,
};

// One rendered answer option: `name` is the caller-facing answer key (a choice question's
// criteria key, or a score level's stringified index "0".."n-1", or "false"/"true" for noul) and
// `description` is the optional rubric text shown to the model for that option.
struct SystemOneOption {
    std::string name;
    std::string description; // empty means "no description given"
};

struct SystemOneQuestion {
    std::string id;
    SystemOneQuestionType type = SystemOneQuestionType::Noul;
    std::string instructions;
    // Rendered in prompt/label order: noul is always [false, true]; choice/score follow the
    // caller's criteria order. `name` is what the wire response keys probabilities/legend by;
    // the single-letter-or-digit prompt label is assigned positionally at prompt-build time.
    std::vector<SystemOneOption> options;
};

} // namespace ninfer::serve
