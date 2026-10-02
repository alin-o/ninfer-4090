#pragma once

// Shared math for turning a small set of full-vocabulary log p(candidate|prompt) values (as
// produced by Engine next-token scoring, ExecutionOptions::score_candidates) into a restricted
// probability distribution over just that candidate set. Used by both /v1/score
// (restricted_probability, candidate_mass) and SystemOne (answer probabilities/confidence).

#include <cstddef>
#include <vector>

namespace ninfer::serve {

// Numerically stable double-precision softmax over natural-log probabilities that are not
// themselves normalized over the candidate set (each is a full-vocabulary conditional
// log-probability). Returns p_i = exp(logprobs[i]) / sum_j exp(logprobs[j]), computed via a
// max-subtraction log-sum-exp so neither overflow nor underflow loses precision for the very
// negative logprobs a large vocabulary produces.
[[nodiscard]] std::vector<double> restricted_softmax(const std::vector<float>& logprobs);

// sum_i exp(logprobs[i]): the total full-vocabulary probability mass captured by the candidate
// set, computed in double precision directly (no restricted-softmax renormalization).
[[nodiscard]] double candidate_probability_mass(const std::vector<float>& logprobs);

// argmax index, or logprobs.size() if empty.
[[nodiscard]] std::size_t argmax_index(const std::vector<double>& probabilities);

// Confidence formula shared by SystemOne choice/score answers:
// clamp((N*peak - 1) / (N - 1), 0, 1) for N candidates and peak = max probability. N<=1 is
// defined as full confidence (a single-option question has no ambiguity to express).
[[nodiscard]] double confidence_from_peak(double peak, std::size_t candidate_count);

} // namespace ninfer::serve
