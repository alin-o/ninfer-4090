#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::runtime::testing {

// Deterministic CUDA delay used only by lifecycle regressions. When installed, Program invokes
// this submission hook on its existing transfer stream at the real producer/pin boundary.
// Production never installs one; the hook owns no Program store or logical catalog state.
struct ContextTransferTestGate {
    void* context                                                       = nullptr;
    cudaError_t (*enqueue)(void* context, cudaStream_t stream) noexcept = nullptr;
};

// Exact producer-side Host footprint for the most recent retained-session snapshot attempt.
// `resident_bytes` includes the complete pinned staging image and the complete pageable assembly
// image that coexist while the bounded writer consumes the completed transfer.
struct SnapshotHostAccounting {
    std::size_t metadata_bytes   = 0;
    std::size_t state_bytes      = 0;
    std::size_t main_kv_bytes    = 0;
    std::size_t backend_kv_bytes = 0;
    std::size_t transfer_bytes   = 0;
    std::size_t staging_bytes    = 0;
    std::size_t resident_bytes   = 0;
};

struct SnapshotShutdownCleanup {
    std::uint64_t observations         = 0;
    std::uint64_t catalog_owners       = 0;
    std::uint64_t device_state_slots   = 0;
    std::uint64_t host_state_slots     = 0;
    std::uint64_t device_main_kv_pages = 0;
    std::uint64_t device_backend_pages = 0;
    std::uint64_t host_kv_bytes        = 0;
};

enum class SharedSnapshotImportStage : std::uint8_t {
    StateAllocated,
    MainKvAllocated,
    BeforeCatalogPublication,
};

// Deterministic adoption checkpoints for real-Program rollback regressions. Production never
// installs this hook. The callback may request cancellation or throw an injected exception.
struct SharedSnapshotImportTestGate {
    void* context                                                      = nullptr;
    void (*checkpoint)(void* context, SharedSnapshotImportStage stage) = nullptr;
};

void install_shared_snapshot_import_gate(const SharedSnapshotImportTestGate* gate) noexcept;
void clear_shared_snapshot_import_gate() noexcept;
void shared_snapshot_import_checkpoint(SharedSnapshotImportStage stage);

enum class SharedSnapshotExportStage : std::uint8_t {
    StatePinnedBeforeRegistration,
    KvPinnedBeforeRegistration,
};

// Deterministic source-registration checkpoints for real-Program export rollback regressions.
// The callback runs after the physical source is pinned but before the pin is appended to the
// transfer settlement's cleanup vector, so an injected bad_alloc exercises the actual ownership
// gap that the guarded registration must close.
struct SharedSnapshotExportTestGate {
    void* context                                                      = nullptr;
    void (*checkpoint)(void* context, SharedSnapshotExportStage stage) = nullptr;
};

void install_shared_snapshot_export_gate(const SharedSnapshotExportTestGate* gate) noexcept;
void clear_shared_snapshot_export_gate() noexcept;
void shared_snapshot_export_checkpoint(SharedSnapshotExportStage stage);
void note_shared_snapshot_export_pin_acquired() noexcept;
void note_shared_snapshot_export_pin_released() noexcept;
[[nodiscard]] std::uint64_t shared_snapshot_export_pinned_sources() noexcept;

void install_snapshot_transfer_gate(const ContextTransferTestGate* gate) noexcept;
void clear_snapshot_transfer_gate() noexcept;
[[nodiscard]] const ContextTransferTestGate* snapshot_transfer_gate() noexcept;
void note_snapshot_transfer_gate_wait() noexcept;
[[nodiscard]] std::uint64_t snapshot_transfer_gate_waits() noexcept;
void note_snapshot_transfer_ownership_acquired(std::size_t backing_bytes,
                                               std::size_t pinned_sources) noexcept;
void note_snapshot_transfer_pins_released(std::size_t pinned_sources) noexcept;
void note_snapshot_transfer_backing_released(std::size_t backing_bytes) noexcept;
[[nodiscard]] std::uint64_t snapshot_transfer_live_settlements() noexcept;
[[nodiscard]] std::uint64_t snapshot_transfer_live_backing_bytes() noexcept;
[[nodiscard]] std::uint64_t snapshot_transfer_pinned_sources() noexcept;
void note_snapshot_host_accounting(SnapshotHostAccounting accounting) noexcept;
[[nodiscard]] SnapshotHostAccounting snapshot_host_accounting() noexcept;
void note_materialization_submitted_cancellation() noexcept;
[[nodiscard]] std::uint64_t materialization_submitted_cancellations() noexcept;
void note_snapshot_shutdown_cleanup(SnapshotShutdownCleanup cleanup) noexcept;
[[nodiscard]] SnapshotShutdownCleanup snapshot_shutdown_cleanup() noexcept;

void install_active_capture_transfer_gate(const ContextTransferTestGate* gate) noexcept;
void clear_active_capture_transfer_gate() noexcept;
[[nodiscard]] const ContextTransferTestGate* active_capture_transfer_gate() noexcept;
void note_active_capture_transfer_gate_wait() noexcept;
[[nodiscard]] std::uint64_t active_capture_transfer_gate_waits() noexcept;
void note_active_capture_submitted_cancellation() noexcept;
[[nodiscard]] std::uint64_t active_capture_submitted_cancellations() noexcept;

} // namespace ninfer::runtime::testing
