// Toolchain smoke test: prove nvcc + MSVC + torch extension + sm_120 all work
// together before investing in a real kernel.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void add_one_kernel(const float* __restrict__ in,
                               float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

torch::Tensor add_one(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                        out.data_ptr<float>(), n);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "add one (CUDA)");
}
