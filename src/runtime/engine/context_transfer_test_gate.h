#pragma once

#include <cuda_runtime.h>

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

void install_active_capture_transfer_gate(const ContextTransferTestGate* gate) noexcept;
void clear_active_capture_transfer_gate() noexcept;
[[nodiscard]] const ContextTransferTestGate* active_capture_transfer_gate() noexcept;
void note_active_capture_transfer_gate_wait() noexcept;
[[nodiscard]] std::uint64_t active_capture_transfer_gate_waits() noexcept;

} // namespace ninfer::runtime::testing
