#pragma once

// Shared wire-neutral "named checkpoint" reference for /v1/score, /v1/systemone, and
// /v1/systemone/batch: {"slot": N, "session_digest": "..."}, the same slot id and digest /slots
// and chat-completion responses' id_slot/session_digest expose.
//
// A checkpoint extends a retained slot's session with a read-only continuation: the request's
// prompt becomes GenerationService::slot_tokens(slot, session_digest) (the slot's confirmed
// token ledger) followed by continuation tokens built from the request body, submitted as an
// ordinary scoring request. ExecutionOptions::score_candidates is itself read-only (never
// publishes, never consumes/advances a matched slot), so the checkpoint slot is only ever read,
// never mutated -- a repeated checkpoint score never changes what /slots reports for it.

#include <cstdint>
#include <string>

namespace ninfer::serve {

struct CheckpointRef {
    bool present = false;
    std::uint32_t slot = 0;
    std::string session_digest;
};

} // namespace ninfer::serve
