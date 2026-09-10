#include "runtime/engine/context_transfer_test_gate.h"

#include <atomic>

namespace ninfer::runtime::testing {
namespace {

std::atomic<const ContextTransferTestGate*> gate{nullptr};
std::atomic<std::uint64_t> waits{0};
std::atomic<const ContextTransferTestGate*> active_capture_gate{nullptr};
std::atomic<std::uint64_t> active_capture_waits{0};

} // namespace

void install_snapshot_transfer_gate(const ContextTransferTestGate* installed) noexcept {
    waits.store(0, std::memory_order_release);
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

void install_active_capture_transfer_gate(const ContextTransferTestGate* installed) noexcept {
    active_capture_waits.store(0, std::memory_order_release);
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

} // namespace ninfer::runtime::testing
