#pragma once

// Maps a ScoreResult (full-vocabulary log p(label|prompt) for a question's candidate labels) onto
// one SystemOne wire answer object. Pure math/JSON-shape layer -- no Engine dependency, so it is
// directly unit-testable.

#include "serve/systemone.h"

#include <nlohmann/json.hpp>

#include <vector>

namespace ninfer::serve {

// `candidate_logprobs` must be index-aligned with `question.options` (the order
// build_systemone_prompt filled candidate_tokens_out in). Confidence uses NInfer's own formula
// (see restricted_softmax.h's confidence_from_peak), not either of laya's two different and
// unrelated confidence quantities -- see systemone.h's header comment for why that is original
// rather than ported.
[[nodiscard]] nlohmann::ordered_json
make_systemone_answer(const SystemOneQuestion& question, const std::vector<float>& candidate_logprobs);

} // namespace ninfer::serve
