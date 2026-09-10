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

void install_active_capture_transfer_gate(const ContextTransferTestGate* gate) noexcept;
void clear_active_capture_transfer_gate() noexcept;
[[nodiscard]] const ContextTransferTestGate* active_capture_transfer_gate() noexcept;
void note_active_capture_transfer_gate_wait() noexcept;
[[nodiscard]] std::uint64_t active_capture_transfer_gate_waits() noexcept;
void note_active_capture_submitted_cancellation() noexcept;
[[nodiscard]] std::uint64_t active_capture_submitted_cancellations() noexcept;

} // namespace ninfer::runtime::testing
