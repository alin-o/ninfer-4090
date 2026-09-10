#include <cuda_runtime.h>

namespace {

__global__ void context_transfer_failure_kernel() { __trap(); }

} // namespace

cudaError_t launch_context_transfer_failure(cudaStream_t stream) noexcept {
    context_transfer_failure_kernel<<<1, 1, 0, stream>>>();
    return cudaPeekAtLastError();
}
