#include "serve/systemone_request.h"

#include <nlohmann/json.hpp>

#include <unordered_set>

namespace ninfer::serve {

void systemone_error(int status, std::string message) {
    ApiError error;
    error.status  = status;
    error.type    = status >= 500 ? "server_error" : "invalid_request_error";
    error.message = std::move(message);
    throw ApiException(std::move(error));
}

namespace {

// Python's `len(str)` counts Unicode code points; this is the same count for any UTF-8 encoded
// text (every code point contributes exactly one byte that is not a UTF-8 continuation byte).
std::size_t utf8_length(std::string_view text) noexcept {
    std::size_t count = 0;
    for (const unsigned char byte : text) {
        if ((byte & 0xC0) != 0x80) { ++count; }
    }
    return count;
}

std::string describe_value(const RequestJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return value.dump();
}

[[nodiscard]] bool fits_slot_id(const RequestJson& value) noexcept {
    if (!value.is_number_integer() && !value.is_number_unsigned()) { return false; }
    return value.get<std::int64_t>() >= 0;
}

// Absent/null "checkpoint" yields CheckpointRef{} (not present). See checkpoint_ref.h.
CheckpointRef parse_checkpoint(const RequestJson& body) {
    CheckpointRef checkpoint;
    if (!body.contains("checkpoint") || body.at("checkpoint").is_null()) { return checkpoint; }
    const RequestJson& value = body.at("checkpoint");
    if (!value.is_object()) {
        systemone_error(400, "checkpoint must be an object with 'slot' and 'session_digest'");
    }
    if (!value.contains("slot") || !fits_slot_id(value.at("slot"))) {
        systemone_error(400, "checkpoint.slot must be a non-negative integer");
    }
    if (!value.contains("session_digest") || !value.at("session_digest").is_string() ||
        value.at("session_digest").get<std::string>().empty()) {
        systemone_error(400, "checkpoint.session_digest must be a non-empty string");
    }
    checkpoint.present        = true;
    checkpoint.slot           = static_cast<std::uint32_t>(value.at("slot").get<std::uint64_t>());
    checkpoint.session_digest = value.at("session_digest").get<std::string>();
    return checkpoint;
}

SystemOneQuestion parse_one_question(const std::string& id, const RequestJson& qdef) {
    if (id.empty()) { systemone_error(422, "question id must not be empty"); }
    if (!qdef.is_object()) {
        systemone_error(422, "question " + id + ": definition must be an object");
    }

    SystemOneQuestion question;
    question.id = id;

    if (!qdef.contains("type") || !qdef.at("type").is_string()) {
        systemone_error(422, "question " + id + ": 'type' must be one of noul, choice, score");
    }
    const std::string type = qdef.at("type").get<std::string>();
    if (type == "noul") {
        question.type = SystemOneQuestionType::Noul;
    } else if (type == "choice") {
        question.type = SystemOneQuestionType::Choice;
    } else if (type == "score") {
        question.type = SystemOneQuestionType::Score;
    } else {
        systemone_error(422, "question " + id + ": unknown type '" + type +
                                 "'; use one of noul, choice, score");
    }

    if (!qdef.contains("instructions") || !qdef.at("instructions").is_string() ||
        qdef.at("instructions").get<std::string>().empty()) {
        systemone_error(422, "question " + id +
                                 ": 'instructions' must be a non-empty string naming the text the "
                                 "model should answer");
    }
    question.instructions = qdef.at("instructions").get<std::string>();

    const RequestJson* criteria = qdef.contains("criteria") ? &qdef.at("criteria") : nullptr;

    if (question.type == SystemOneQuestionType::Choice) {
        if (criteria == nullptr || (!criteria->is_object() && !criteria->is_array())) {
            systemone_error(422, "question " + id +
                                     ": a choice question takes 'criteria' as an object of "
                                     "label -> description, or an array of labels");
        }
        std::unordered_set<std::string> seen_names;
        if (criteria->is_object()) {
            for (const auto& [key, value] : criteria->items()) {
                if (key.empty()) {
                    systemone_error(422, "question " + id + ": a choice label must not be empty");
                }
                if (!seen_names.insert(key).second) {
                    systemone_error(422, "question " + id + ": choice label '" + key +
                                             "' repeats; labels are the answer keys");
                }
                SystemOneOption option;
                option.name        = key;
                option.description = value.is_null() ? std::string() : describe_value(value);
                question.options.push_back(std::move(option));
            }
        } else {
            for (const RequestJson& item : *criteria) {
                if (!item.is_string() || item.get<std::string>().empty()) {
                    systemone_error(422, "question " + id +
                                             ": choice labels given as an array must be "
                                             "non-empty strings");
                }
                const std::string label = item.get<std::string>();
                if (!seen_names.insert(label).second) {
                    systemone_error(422, "question " + id + ": choice label '" + label +
                                             "' repeats; labels are the answer keys");
                }
                question.options.push_back(SystemOneOption{label, std::string()});
            }
        }
        if (question.options.empty()) {
            systemone_error(422, "question " + id + ": a choice question needs at least one option");
        }
        if (question.options.size() > kSystemOneMaxChoiceOptions) {
            systemone_error(413, "question " + id + ": too many choice options (" +
                                     std::to_string(question.options.size()) + " > " +
                                     std::to_string(kSystemOneMaxChoiceOptions) + ")");
        }
    } else if (question.type == SystemOneQuestionType::Score) {
        if (criteria == nullptr || !criteria->is_array()) {
            systemone_error(422, "question " + id +
                                     ": a score question takes 'criteria' as an array of level "
                                     "descriptions, index 0 first");
        }
        std::size_t index = 0;
        for (const RequestJson& item : *criteria) {
            if (!item.is_string()) {
                systemone_error(422, "question " + id + ": score level " + std::to_string(index) +
                                         " must be a string description");
            }
            question.options.push_back(SystemOneOption{std::to_string(index), item.get<std::string>()});
            ++index;
        }
        if (question.options.empty()) {
            systemone_error(422, "question " + id + ": a score question needs at least one level");
        }
        if (question.options.size() > kSystemOneMaxScoreLevels) {
            systemone_error(413, "question " + id + ": too many score levels (" +
                                     std::to_string(question.options.size()) + " > " +
                                     std::to_string(kSystemOneMaxScoreLevels) + ")");
        }
    } else { // Noul
        std::string false_description;
        std::string true_description;
        if (criteria != nullptr && !criteria->is_null()) {
            if (!criteria->is_object()) {
                systemone_error(422, "question " + id +
                                         ": a noul question takes 'criteria' as an object with "
                                         "optional 'true'/'false' descriptions, or omits it");
            }
            for (const auto& [key, value] : criteria->items()) {
                if (key != "true" && key != "false") {
                    systemone_error(422, "question " + id +
                                             ": a noul question's 'criteria' may only be keyed "
                                             "'true'/'false'; got '" +
                                             key + "'");
                }
                const std::string description = value.is_null() ? std::string() : describe_value(value);
                (key == "true" ? true_description : false_description) = description;
            }
        }
        // Fixed label order [false, true]; the wire-facing label text is NInfer's own default
        // ("Yes"/"No") assigned in the prompt layer, not here -- `name` is the answer key.
        question.options.push_back(SystemOneOption{"false", std::move(false_description)});
        question.options.push_back(SystemOneOption{"true", std::move(true_description)});
    }

    return question;
}

} // namespace

std::string serialize_systemone_state(const RequestJson& raw_state) {
    const RequestJson* state = &raw_state;
    if (state->is_object() && state->size() == 1 && state->contains("body")) {
        state = &state->at("body");
    }
    if (state->is_string()) { return state->get<std::string>(); }
    return state->dump();
}

std::vector<SystemOneQuestion> parse_systemone_questions(const RequestJson& body) {
    if (!body.contains("questions") || !body.at("questions").is_object()) {
        systemone_error(400, "'questions' must be an object");
    }
    const RequestJson& questions = body.at("questions");
    if (questions.size() > kSystemOneMaxQuestions) {
        systemone_error(413, "too many questions (" + std::to_string(questions.size()) + " > " +
                                 std::to_string(kSystemOneMaxQuestions) + ")");
    }
    std::vector<SystemOneQuestion> parsed;
    parsed.reserve(questions.size());
    for (const auto& [id, qdef] : questions.items()) { parsed.push_back(parse_one_question(id, qdef)); }
    return parsed;
}

SystemOneRequest parse_systemone_request(const RequestJson& body) {
    if (!body.is_object()) {
        systemone_error(400, "request body must be an object with 'state' and 'questions' fields");
    }

    SystemOneRequest request;
    if (body.contains("model") && !body.at("model").is_null()) {
        if (!body.at("model").is_string()) { systemone_error(400, "model must be a string"); }
        request.model = body.at("model").get<std::string>();
    }

    request.checkpoint = parse_checkpoint(body);

    if (!body.contains("state") || body.at("state").is_null()) {
        if (!request.checkpoint.present) {
            systemone_error(400, "'state' is required unless 'checkpoint' is given");
        }
        request.has_state = false;
    } else {
        request.state      = serialize_systemone_state(body.at("state"));
        request.has_state  = true;
        if (utf8_length(request.state) > kSystemOneMaxStateChars) {
            systemone_error(413, "state too large (" + std::to_string(utf8_length(request.state)) +
                                     " > " + std::to_string(kSystemOneMaxStateChars) + " chars)");
        }
    }

    request.questions = parse_systemone_questions(body);
    return request;
}

SystemOneBatchRequest parse_systemone_batch_request(const RequestJson& body) {
    if (!body.is_object()) {
        systemone_error(400, "request body must be an object with 'states' and 'questions' fields");
    }

    SystemOneBatchRequest request;
    if (body.contains("model") && !body.at("model").is_null()) {
        if (!body.at("model").is_string()) { systemone_error(400, "model must be a string"); }
        request.model = body.at("model").get<std::string>();
    }

    request.checkpoint = parse_checkpoint(body);

    if (!body.contains("states") || !body.at("states").is_array() || body.at("states").empty()) {
        systemone_error(400, "'states' must be a non-empty array");
    }
    const RequestJson& states = body.at("states");
    if (states.size() > kSystemOneMaxBatchStates) {
        systemone_error(413, "too many states in batch (" + std::to_string(states.size()) + " > " +
                                 std::to_string(kSystemOneMaxBatchStates) + ")");
    }
    request.states.reserve(states.size());
    for (const RequestJson& state : states) {
        if (state.is_null()) { systemone_error(400, "'states' entries must not be null"); }
        std::string serialized = serialize_systemone_state(state);
        if (utf8_length(serialized) > kSystemOneMaxStateChars) {
            systemone_error(413, "state too large (" + std::to_string(utf8_length(serialized)) +
                                     " > " + std::to_string(kSystemOneMaxStateChars) + " chars)");
        }
        request.states.push_back(std::move(serialized));
    }

    request.questions = parse_systemone_questions(body);
    return request;
}

} // namespace ninfer::serve
