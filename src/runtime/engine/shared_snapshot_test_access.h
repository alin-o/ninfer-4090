#pragma once

#include "ninfer/engine.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h"

#include <cstdint>
#include <span>
#include <utility>

namespace ninfer::runtime::testing {

struct SharedSnapshotImportObservation {
    std::uint32_t disposition                 = 0;
    std::uint32_t slot                        = 0;
    std::uint32_t frontier                    = 0;
    std::uint32_t main_frontier               = 0;
    std::uint32_t backend_frontier            = 0;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
};

// Internal real-Program regression access. Production durable code consumes the same EngineCore
// boundary; this adapter only avoids adding a client-visible anchor reference API.
struct SharedSnapshotTestAccess {
    [[nodiscard]] static std::pair<std::uint32_t, targets::qwen3_6::RetainedSessionSnapshot>
    export_first_durable(Engine& engine);
    [[nodiscard]] static targets::qwen3_6::RetainedSessionSnapshot export_slot(Engine& engine,
                                                                               std::uint32_t slot);
    [[nodiscard]] static SharedSnapshotImportObservation
    import(Engine& engine, std::span<const std::uint8_t> bytes);
    [[nodiscard]] static std::uint32_t import_cancelled(Engine& engine,
                                                        std::span<const std::uint8_t> bytes);
};

} // namespace ninfer::runtime::testing
