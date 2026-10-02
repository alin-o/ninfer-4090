#include "serve/systemone_answer.h"

#include "serve/restricted_softmax.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::serve {

nlohmann::ordered_json make_systemone_answer(const SystemOneQuestion& question,
                                             const std::vector<float>& candidate_logprobs) {
    if (candidate_logprobs.size() != question.options.size()) {
        throw std::runtime_error("SystemOne answer: candidate count does not match option count");
    }
    const std::vector<double> probabilities = restricted_softmax(candidate_logprobs);
    const std::size_t peak_index            = argmax_index(probabilities);
    const double peak                       = probabilities[peak_index];
    const double confidence     = confidence_from_peak(peak, probabilities.size());

    nlohmann::ordered_json answer;
    switch (question.type) {
    case SystemOneQuestionType::Noul: {
        // options[0] = "false", options[1] = "true" (fixed order from parse_systemone_questions).
        const double p_yes = probabilities[1];
        answer["type"]       = "noul";
        answer["noul"]       = p_yes;
        answer["confidence"] = confidence_from_peak(std::max(p_yes, 1.0 - p_yes), 2);
        break;
    }
    case SystemOneQuestionType::Choice: {
        nlohmann::ordered_json option_probabilities = nlohmann::ordered_json::object();
        for (std::size_t i = 0; i < question.options.size(); ++i) {
            option_probabilities[question.options[i].name] = probabilities[i];
        }
        answer["type"]          = "choice";
        answer["choice"]        = question.options[peak_index].name;
        answer["probabilities"] = std::move(option_probabilities);
        answer["confidence"]    = confidence;
        break;
    }
    case SystemOneQuestionType::Score: {
        double expected_score = 0.0;
        nlohmann::ordered_json legend        = nlohmann::ordered_json::object();
        nlohmann::ordered_json option_probabilities = nlohmann::ordered_json::object();
        for (std::size_t i = 0; i < question.options.size(); ++i) {
            expected_score += static_cast<double>(i) * probabilities[i];
            legend[question.options[i].name]               = question.options[i].description;
            option_probabilities[question.options[i].name] = probabilities[i];
        }
        answer["type"]          = "score";
        answer["score"]         = expected_score;
        answer["legend"]        = std::move(legend);
        answer["probabilities"] = std::move(option_probabilities);
        answer["confidence"]    = confidence;
        break;
    }
    }
    return answer;
}

} // namespace ninfer::serve
