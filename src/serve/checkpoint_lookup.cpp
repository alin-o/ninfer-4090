#include "serve/checkpoint_lookup.h"

#include "serve/request_validation.h"

#include <stdexcept>
#include <utility>

namespace ninfer::serve {

std::vector<ninfer::TokenId> resolve_checkpoint_ledger(const GenerationService& service,
                                                       const CheckpointRef& checkpoint) {
    try {
        return service.slot_tokens(checkpoint.slot, checkpoint.session_digest);
    } catch (const ninfer::SlotSessionMismatch& mismatch) {
        ApiError error;
        error.status  = 409;
        error.type    = "invalid_request_error";
        error.code    = "slot_session_mismatch";
        error.message = mismatch.what();
        throw ApiException(std::move(error));
    } catch (const ninfer::RequestError& busy) {
        ApiError error;
        error.status  = 409;
        error.type    = "invalid_request_error";
        error.code    = "slot_busy";
        error.message = busy.what();
        throw ApiException(std::move(error));
    } catch (const std::invalid_argument& invalid) {
        bad_request(std::string("checkpoint: ") + invalid.what(), "checkpoint",
                   "invalid_checkpoint_slot");
    }
}

} // namespace ninfer::serve
