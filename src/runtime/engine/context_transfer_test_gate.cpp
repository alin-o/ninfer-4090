#include "runtime/engine/context_transfer_test_gate.h"

#include <atomic>

namespace ninfer::runtime::testing {
namespace {

std::atomic<const ContextTransferTestGate*> gate{nullptr};
std::atomic<std::uint64_t> waits{0};
std::atomic<std::uint64_t> live_settlements{0};
std::atomic<std::uint64_t> live_backing_bytes{0};
std::atomic<std::uint64_t> pinned_sources{0};
std::atomic<std::size_t> snapshot_metadata_bytes{0};
std::atomic<std::size_t> snapshot_state_bytes{0};
std::atomic<std::size_t> snapshot_main_kv_bytes{0};
std::atomic<std::size_t> snapshot_backend_kv_bytes{0};
std::atomic<std::size_t> snapshot_transfer_bytes{0};
std::atomic<std::size_t> snapshot_staging_bytes{0};
std::atomic<std::size_t> snapshot_resident_bytes{0};
std::atomic<std::uint64_t> materialization_cancellations{0};
std::atomic<std::uint64_t> shutdown_cleanup_observations{0};
std::atomic<std::uint64_t> shutdown_catalog_owners{0};
std::atomic<std::uint64_t> shutdown_device_state_slots{0};
std::atomic<std::uint64_t> shutdown_host_state_slots{0};
std::atomic<std::uint64_t> shutdown_device_main_kv_pages{0};
std::atomic<std::uint64_t> shutdown_device_backend_pages{0};
std::atomic<std::uint64_t> shutdown_host_kv_bytes{0};
std::atomic<const ContextTransferTestGate*> active_capture_gate{nullptr};
std::atomic<std::uint64_t> active_capture_waits{0};
std::atomic<std::uint64_t> active_capture_cancellations{0};
std::atomic<const SharedSnapshotImportTestGate*> shared_import_gate{nullptr};

} // namespace

void install_shared_snapshot_import_gate(const SharedSnapshotImportTestGate* installed) noexcept {
    shared_import_gate.store(installed, std::memory_order_release);
}

void clear_shared_snapshot_import_gate() noexcept {
    shared_import_gate.store(nullptr, std::memory_order_release);
}

void shared_snapshot_import_checkpoint(SharedSnapshotImportStage stage) {
    const SharedSnapshotImportTestGate* installed =
        shared_import_gate.load(std::memory_order_acquire);
    if (installed != nullptr && installed->checkpoint != nullptr) {
        installed->checkpoint(installed->context, stage);
    }
}

void install_snapshot_transfer_gate(const ContextTransferTestGate* installed) noexcept {
    waits.store(0, std::memory_order_release);
    live_settlements.store(0, std::memory_order_release);
    live_backing_bytes.store(0, std::memory_order_release);
    pinned_sources.store(0, std::memory_order_release);
    snapshot_metadata_bytes.store(0, std::memory_order_release);
    snapshot_state_bytes.store(0, std::memory_order_release);
    snapshot_main_kv_bytes.store(0, std::memory_order_release);
    snapshot_backend_kv_bytes.store(0, std::memory_order_release);
    snapshot_transfer_bytes.store(0, std::memory_order_release);
    snapshot_staging_bytes.store(0, std::memory_order_release);
    snapshot_resident_bytes.store(0, std::memory_order_release);
    materialization_cancellations.store(0, std::memory_order_release);
    shutdown_cleanup_observations.store(0, std::memory_order_release);
    shutdown_catalog_owners.store(0, std::memory_order_release);
    shutdown_device_state_slots.store(0, std::memory_order_release);
    shutdown_host_state_slots.store(0, std::memory_order_release);
    shutdown_device_main_kv_pages.store(0, std::memory_order_release);
    shutdown_device_backend_pages.store(0, std::memory_order_release);
    shutdown_host_kv_bytes.store(0, std::memory_order_release);
    gate.store(installed, std::memory_order_release);
}

void clear_snapshot_transfer_gate() noexcept { gate.store(nullptr, std::memory_order_release); }

const ContextTransferTestGate* snapshot_transfer_gate() noexcept {
    return gate.load(std::memory_order_acquire);
}

void note_snapshot_transfer_gate_wait() noexcept { waits.fetch_add(1, std::memory_order_acq_rel); }

std::uint64_t snapshot_transfer_gate_waits() noexcept {
    return waits.load(std::memory_order_acquire);
}

void note_snapshot_transfer_ownership_acquired(std::size_t backing_bytes,
                                               std::size_t source_count) noexcept {
    live_settlements.fetch_add(1, std::memory_order_acq_rel);
    live_backing_bytes.fetch_add(backing_bytes, std::memory_order_acq_rel);
    pinned_sources.fetch_add(source_count, std::memory_order_acq_rel);
}

void note_snapshot_transfer_pins_released(std::size_t source_count) noexcept {
    pinned_sources.fetch_sub(source_count, std::memory_order_acq_rel);
}

void note_snapshot_transfer_backing_released(std::size_t backing_bytes) noexcept {
    live_backing_bytes.fetch_sub(backing_bytes, std::memory_order_acq_rel);
    live_settlements.fetch_sub(1, std::memory_order_acq_rel);
}

std::uint64_t snapshot_transfer_live_settlements() noexcept {
    return live_settlements.load(std::memory_order_acquire);
}

std::uint64_t snapshot_transfer_live_backing_bytes() noexcept {
    return live_backing_bytes.load(std::memory_order_acquire);
}

std::uint64_t snapshot_transfer_pinned_sources() noexcept {
    return pinned_sources.load(std::memory_order_acquire);
}

void note_snapshot_host_accounting(SnapshotHostAccounting accounting) noexcept {
    snapshot_metadata_bytes.store(accounting.metadata_bytes, std::memory_order_relaxed);
    snapshot_state_bytes.store(accounting.state_bytes, std::memory_order_relaxed);
    snapshot_main_kv_bytes.store(accounting.main_kv_bytes, std::memory_order_relaxed);
    snapshot_backend_kv_bytes.store(accounting.backend_kv_bytes, std::memory_order_relaxed);
    snapshot_transfer_bytes.store(accounting.transfer_bytes, std::memory_order_relaxed);
    snapshot_staging_bytes.store(accounting.staging_bytes, std::memory_order_relaxed);
    // Publish readiness last so an acquiring reader observes one complete accounting record.
    snapshot_resident_bytes.store(accounting.resident_bytes, std::memory_order_release);
}

SnapshotHostAccounting snapshot_host_accounting() noexcept {
    SnapshotHostAccounting accounting;
    accounting.resident_bytes   = snapshot_resident_bytes.load(std::memory_order_acquire);
    accounting.metadata_bytes   = snapshot_metadata_bytes.load(std::memory_order_relaxed);
    accounting.state_bytes      = snapshot_state_bytes.load(std::memory_order_relaxed);
    accounting.main_kv_bytes    = snapshot_main_kv_bytes.load(std::memory_order_relaxed);
    accounting.backend_kv_bytes = snapshot_backend_kv_bytes.load(std::memory_order_relaxed);
    accounting.transfer_bytes   = snapshot_transfer_bytes.load(std::memory_order_relaxed);
    accounting.staging_bytes    = snapshot_staging_bytes.load(std::memory_order_relaxed);
    return accounting;
}

void note_materialization_submitted_cancellation() noexcept {
    materialization_cancellations.fetch_add(1, std::memory_order_acq_rel);
}

std::uint64_t materialization_submitted_cancellations() noexcept {
    return materialization_cancellations.load(std::memory_order_acquire);
}

void note_snapshot_shutdown_cleanup(SnapshotShutdownCleanup cleanup) noexcept {
    shutdown_catalog_owners.store(cleanup.catalog_owners, std::memory_order_relaxed);
    shutdown_device_state_slots.store(cleanup.device_state_slots, std::memory_order_relaxed);
    shutdown_host_state_slots.store(cleanup.host_state_slots, std::memory_order_relaxed);
    shutdown_device_main_kv_pages.store(cleanup.device_main_kv_pages, std::memory_order_relaxed);
    shutdown_device_backend_pages.store(cleanup.device_backend_pages, std::memory_order_relaxed);
    shutdown_host_kv_bytes.store(cleanup.host_kv_bytes, std::memory_order_relaxed);
    shutdown_cleanup_observations.fetch_add(1, std::memory_order_release);
}

SnapshotShutdownCleanup snapshot_shutdown_cleanup() noexcept {
    SnapshotShutdownCleanup cleanup;
    cleanup.observations         = shutdown_cleanup_observations.load(std::memory_order_acquire);
    cleanup.catalog_owners       = shutdown_catalog_owners.load(std::memory_order_relaxed);
    cleanup.device_state_slots   = shutdown_device_state_slots.load(std::memory_order_relaxed);
    cleanup.host_state_slots     = shutdown_host_state_slots.load(std::memory_order_relaxed);
    cleanup.device_main_kv_pages = shutdown_device_main_kv_pages.load(std::memory_order_relaxed);
    cleanup.device_backend_pages = shutdown_device_backend_pages.load(std::memory_order_relaxed);
    cleanup.host_kv_bytes        = shutdown_host_kv_bytes.load(std::memory_order_relaxed);
    return cleanup;
}

void install_active_capture_transfer_gate(const ContextTransferTestGate* installed) noexcept {
    active_capture_waits.store(0, std::memory_order_release);
    active_capture_cancellations.store(0, std::memory_order_release);
    active_capture_gate.store(installed, std::memory_order_release);
}

void clear_active_capture_transfer_gate() noexcept {
    active_capture_gate.store(nullptr, std::memory_order_release);
}

const ContextTransferTestGate* active_capture_transfer_gate() noexcept {
    return active_capture_gate.load(std::memory_order_acquire);
}

void note_active_capture_transfer_gate_wait() noexcept {
    active_capture_waits.fetch_add(1, std::memory_order_acq_rel);
}

std::uint64_t active_capture_transfer_gate_waits() noexcept {
    return active_capture_waits.load(std::memory_order_acquire);
}

void note_active_capture_submitted_cancellation() noexcept {
    active_capture_cancellations.fetch_add(1, std::memory_order_acq_rel);
}

std::uint64_t active_capture_submitted_cancellations() noexcept {
    return active_capture_cancellations.load(std::memory_order_acquire);
}

} // namespace ninfer::runtime::testing
