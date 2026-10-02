#pragma once

// Shared named-checkpoint resolution for /v1/score, /v1/systemone, and /v1/systemone/batch: turns
// a wire-level CheckpointRef into the retained slot's reusable token ledger, with the one error
// mapping shared by every route that touches a slot by id (see src/serve/http_server.cpp's
// handle_slot_action for the /slots precedent this mirrors).

#include "serve/checkpoint_ref.h"
#include "serve/generation_service.h"

#include <ninfer/types.h>

#include <vector>

namespace ninfer::serve {

// Resolves a named checkpoint's reusable token ledger (GenerationService::slot_tokens). Pure read:
// never changes slot occupancy or digest. Maps the Engine's three failure modes the same way the
// /slots routes map Engine::{save,restore,erase}_slot's: ninfer::RequestError(Overloaded) (busy
// lane) -> ApiException(409, "slot_busy"); ninfer::SlotSessionMismatch (stale digest) ->
// ApiException(409, "slot_session_mismatch"); std::invalid_argument (empty/erased slot) ->
// ApiException(400, invalid_request_error) via bad_request. Throws ApiException; callers must
// already be inside a try block that renders ApiException into their route's error contract.
[[nodiscard]] std::vector<ninfer::TokenId>
resolve_checkpoint_ledger(const GenerationService& service, const CheckpointRef& checkpoint);

} // namespace ninfer::serve
