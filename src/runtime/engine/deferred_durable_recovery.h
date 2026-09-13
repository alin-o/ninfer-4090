#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::runtime {

// Immutable filesystem input prepared by Gateway and resolved only after Scheduler has selected
// this request for FIFO admission. Gateway writes only load completion under mutex; the Engine
// worker owns selection and outcome publication, which the consumer reads after settlement.
struct DeferredDurableRecovery {
    struct Candidate {
        std::string content_digest;
        std::uint32_t frontier = 0;

        [[nodiscard]] friend bool operator==(const Candidate&, const Candidate&) noexcept = default;
    };

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::vector<Candidate> available_candidates;
    Candidate candidate;
    // Owned cancellation query handed across the Gateway/Engine boundary. Engine evaluates it at
    // every destructive Program checkpoint while Gateway is blocked in the immutable SSD load.
    CancellationView gateway_cancellation;
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    std::string model_binding;
    std::string filename;
    std::uint64_t manifest_order   = 0;
    std::uint64_t io_elapsed_ns    = 0;
    std::uint64_t serialized_bytes = 0;
    std::uint64_t reservation_id   = 0;
    bool load_requested            = false;
    bool load_completed            = false;

    std::uint32_t frontier               = 0;
    std::uint64_t validation_nanoseconds = 0;
    std::uint64_t adoption_nanoseconds   = 0;
    bool completed                       = false;
    bool loaded_from_ssd                 = false;
    bool warm_available                  = false;
    bool invalidate_record               = false;
    bool gateway_settled                 = false;
    std::string fallback_reason;
    std::vector<CheckpointLifecycleFact> lifecycle;
};

} // namespace ninfer::runtime
