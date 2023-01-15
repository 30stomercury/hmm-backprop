#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>


namespace {

template <typename scalar_t>
__global__ void log_matmul_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> a,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> b,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> out,
        const int m, const int p, const int n) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch_size = blockIdx.z;

    // might exceed the sizes of a and b
    if (row < m && col < n) {
        scalar_t val = 0.0;
        scalar_t max = -1e9;
        for (int i = 0; i < p; i++) {
            scalar_t v = a[batch_size][row][i] + b[batch_size][i][col];
            if (v > max) {
                max = v;
            }
        }
        for (int i = 0; i < p; i++) {
            scalar_t v = a[batch_size][row][i] + b[batch_size][i][col];
            val += exp(v - max);
        }
        out[batch_size][row][col] = log(val) + max;
    }
}

} // namespace


/* Sum operation in Log Semiring
 * Matrix multiplication in log space.

Arguments
---------
log_a: (B, m, p) 
log_b: (B, p, n)

Returns
-------
(B, m, n)
*/
torch::Tensor log_matmul_cuda(
        torch::Tensor& a, 
        torch::Tensor& b) {
    const int batch = a.sizes()[0];
    const int m = a.sizes()[1];
    const int p = a.sizes()[2];
    const int n = b.sizes()[2];
    const size_t nthreads = 32;
    const dim3 threads_per_block(nthreads, nthreads, 1);
    const dim3 blocks(m / nthreads + 1, n / nthreads + 1, batch);
    auto options = torch::TensorOptions().dtype(a.dtype()).device(
      torch::kCUDA, a.device().index());
    auto out = torch::empty({batch, m, n}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        a.type(), "log_cuda_matmul", ([&] {
        log_matmul_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            a.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            b.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            m, p, n);
      }));

    return out;
}

