#include "runtime/engine/context_transfer_test_gate.h"

#include <atomic>

namespace ninfer::runtime::testing {
namespace {

std::atomic<const ContextTransferTestGate*> gate{nullptr};
std::atomic<std::uint64_t> waits{0};
std::atomic<std::uint64_t> live_settlements{0};
std::atomic<std::uint64_t> live_backing_bytes{0};
std::atomic<std::uint64_t> pinned_sources{0};
std::atomic<const ContextTransferTestGate*> active_capture_gate{nullptr};
std::atomic<std::uint64_t> active_capture_waits{0};
std::atomic<std::uint64_t> active_capture_cancellations{0};

} // namespace

void install_snapshot_transfer_gate(const ContextTransferTestGate* installed) noexcept {
    waits.store(0, std::memory_order_release);
    live_settlements.store(0, std::memory_order_release);
    live_backing_bytes.store(0, std::memory_order_release);
    pinned_sources.store(0, std::memory_order_release);
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
