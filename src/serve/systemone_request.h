#pragma once

#include "serve/checkpoint_ref.h"
#include "serve/request_json.h"
#include "serve/systemone.h"

#include <string>
#include <vector>

namespace ninfer::serve {

struct SystemOneRequest {
    std::string model; // echoed verbatim; functionally ignored (single resident model)
    // Optional named checkpoint: when present, `state` becomes optional (a question about the
    // checkpointed conversation itself) and the prompt's continuation turn is built on top of
    // GenerationService::slot_tokens(checkpoint.slot, checkpoint.session_digest) instead of the
    // normal [system][state][question] layout.
    CheckpointRef checkpoint;
    bool has_state = false;
    std::string state;  // already serialized to text (serialize_systemone_state); valid iff has_state
    std::vector<SystemOneQuestion> questions; // insertion order preserved
};

struct SystemOneBatchRequest {
    std::string model;
    CheckpointRef checkpoint;
    std::vector<std::string> states;
    std::vector<SystemOneQuestion> questions;
};

// A lone {"body": X} wrapper is unwrapped to X before serialization, matching the brief's state
// normalization rule. A string state passes through unchanged; anything else (object/array)
// becomes compact JSON text, the same "state" the model actually reads.
[[nodiscard]] std::string serialize_systemone_state(const RequestJson& state);

// Parses and validates the `questions` object shared by both routes: type/instructions/criteria
// shape, option-count caps (kSystemOneMaxChoiceOptions / kSystemOneMaxScoreLevels), and the
// kSystemOneMaxQuestions cap. Throws systemone_error(422, ...) for a malformed question and
// systemone_error(413, ...) for an over-limit question count.
[[nodiscard]] std::vector<SystemOneQuestion> parse_systemone_questions(const RequestJson& body);

[[nodiscard]] SystemOneRequest parse_systemone_request(const RequestJson& body);
[[nodiscard]] SystemOneBatchRequest parse_systemone_batch_request(const RequestJson& body);

} // namespace ninfer::serve
