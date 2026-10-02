#include "serve/restricted_softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ninfer::serve {

std::vector<double> restricted_softmax(const std::vector<float>& logprobs) {
    std::vector<double> probabilities(logprobs.size(), 0.0);
    if (logprobs.empty()) { return probabilities; }
    double max_logprob = -std::numeric_limits<double>::infinity();
    for (const float value : logprobs) { max_logprob = std::max(max_logprob, double(value)); }
    double sum = 0.0;
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        const double shifted = std::exp(double(logprobs[i]) - max_logprob);
        probabilities[i]     = shifted;
        sum += shifted;
    }
    if (sum <= 0.0) {
        // Every candidate underflowed to zero relative to the maximum: fall back to a uniform
        // distribution rather than dividing by zero.
        const double uniform = 1.0 / double(logprobs.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
        return probabilities;
    }
    for (double& value : probabilities) { value /= sum; }
    return probabilities;
}

double candidate_probability_mass(const std::vector<float>& logprobs) {
    double sum = 0.0;
    for (const float value : logprobs) { sum += std::exp(double(value)); }
    return sum;
}

std::size_t argmax_index(const std::vector<double>& probabilities) {
    if (probabilities.empty()) { return 0; }
    std::size_t best = 0;
    for (std::size_t i = 1; i < probabilities.size(); ++i) {
        if (probabilities[i] > probabilities[best]) { best = i; }
    }
    return best;
}

double confidence_from_peak(double peak, std::size_t candidate_count) {
    if (candidate_count <= 1) { return 1.0; }
    const double n           = double(candidate_count);
    const double unclamped   = (n * peak - 1.0) / (n - 1.0);
    return std::clamp(unclamped, 0.0, 1.0);
}

} // namespace ninfer::serve
