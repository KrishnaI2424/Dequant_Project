// In-register dequantize + GEMV for packed INT2/4/8 weights (OUTLINE section 6).
//
// The packed format is defined by dequant/packer.py and this file must agree
// with it bit for bit:
//   * weights are [N, K], the reduction runs over K, groups run along K
//   * codes are UNSIGNED; symmetric formats are pre-biased by 2^(b-1)
//   * a uint32 word holds PER_PACK = 32/bits consecutive logical codes
//   * "sequential" puts logical i at bit offset bits*i
//   * "interleaved" puts LOGICAL _INTERLEAVE[bits][s] in SLOT s, which is what
//     makes (w >> bits*i) & <pair mask> yield a pair/quad of CONSECUTIVE
//     logical codes ready for the fp16 magic-number conversion.
//
// Nothing here is newer than sm_53 (half2 intrinsics, __byte_perm, __ldg,
// __shfl_xor_sync), so the sm_75 gencode stays valid.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "dequant_cuda.h"

namespace {

constexpr int ROWS_PER_BLOCK = 4;   // one warp per output row
constexpr int GEMV_THREADS = ROWS_PER_BLOCK * 32;
constexpr int MAX_M = 8;

__device__ __forceinline__ __half2 u32_as_half2(uint32_t u) {
    return __halves2half2(__ushort_as_half(static_cast<unsigned short>(u & 0xFFFFu)),
                          __ushort_as_half(static_cast<unsigned short>(u >> 16)));
}

__device__ __forceinline__ float2 u32_to_float2(uint32_t u) {
    return __half22float2(u32_as_half2(u));
}

// 0x6400 is fp16 1024.0. For q < 1024, (0x6400 | q) is exactly fp16(1024 + q),
// so one __hsub2 turns a packed pair of codes into a pair of exact floats.
__device__ __forceinline__ float2 magic_pair(uint32_t p) {
    const __half2 k1024 = u32_as_half2(0x64006400u);
    return __half22float2(__hsub2(u32_as_half2(p | 0x64006400u), k1024));
}

// v[j] = float(logical code j) - off, in LOGICAL order.
template <int BITS, bool INTER>
__device__ __forceinline__ void unpack_word(uint32_t w, float off, float* v) {
    constexpr int PER_PACK = 32 / BITS;
    constexpr uint32_t QMAX = (1u << BITS) - 1u;

    if constexpr (!INTER) {
#pragma unroll
        for (int i = 0; i < PER_PACK; ++i) {
            v[i] = static_cast<float>((w >> (BITS * i)) & QMAX) - off;
        }
    } else if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = magic_pair((w >> (4 * i)) & 0x000F000Fu);
            v[2 * i] = f.x - off;
            v[2 * i + 1] = f.y - off;
        }
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float2 f = magic_pair((w >> (8 * i)) & 0x00FF00FFu);
            v[2 * i] = f.x - off;
            v[2 * i + 1] = f.y - off;
        }
    } else {  // BITS == 2: one shift+mask yields logical 4i..4i+3, one per byte.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t b = (w >> (2 * i)) & 0x03030303u;
            // 0x4140 -> {b.byte0, 0x64, b.byte1, 0x64}; 0x4342 -> bytes 2 and 3.
            float2 f0 = magic_pair(__byte_perm(b, 0x64646464u, 0x4140));
            float2 f1 = magic_pair(__byte_perm(b, 0x64646464u, 0x4342));
            v[4 * i] = f0.x - off;
            v[4 * i + 1] = f0.y - off;
            v[4 * i + 2] = f1.x - off;
            v[4 * i + 3] = f1.y - off;
        }
    }
}

// sum_i v[i] * x[i] over PER_PACK consecutive fp16 activations. PER_PACK is 4,
// 8 or 16, so this is one uint2 or one/two uint4 loads.
template <int PER_PACK>
__device__ __forceinline__ float dot_x(const __half* __restrict__ p, const float* v) {
    float s = 0.f;
#pragma unroll
    for (int t = 0; t < PER_PACK / 8; ++t) {
        uint4 q = __ldg(reinterpret_cast<const uint4*>(p) + t);
        float2 a = u32_to_float2(q.x);
        float2 b = u32_to_float2(q.y);
        float2 c = u32_to_float2(q.z);
        float2 d = u32_to_float2(q.w);
        s += v[t * 8 + 0] * a.x + v[t * 8 + 1] * a.y
           + v[t * 8 + 2] * b.x + v[t * 8 + 3] * b.y
           + v[t * 8 + 4] * c.x + v[t * 8 + 5] * c.y
           + v[t * 8 + 6] * d.x + v[t * 8 + 7] * d.y;
    }
    if constexpr (PER_PACK == 4) {
        uint2 q = __ldg(reinterpret_cast<const uint2*>(p));
        float2 a = u32_to_float2(q.x);
        float2 b = u32_to_float2(q.y);
        s += v[0] * a.x + v[1] * a.y + v[2] * b.x + v[3] * b.y;
    }
    return s;
}

// ---------------------------------------------------------------- dequantize

template <int BITS, bool SYM, bool INTER>
__global__ void dequantize_kernel(const uint32_t* __restrict__ qweight,
                                  const __half* __restrict__ scales,
                                  const __half* __restrict__ zeros,
                                  __half* __restrict__ out,
                                  int64_t total, int N, int K, int group_size) {
    constexpr int PER_PACK = 32 / BITS;
    constexpr int BIAS = 1 << (BITS - 1);

    int64_t widx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (widx >= total) return;

    const int words_per_row = K / PER_PACK;
    const int n = static_cast<int>(widx / words_per_row);
    const int k0 = static_cast<int>(widx % words_per_row) * PER_PACK;
    const int n_groups = K / group_size;
    const int64_t gidx = static_cast<int64_t>(n) * n_groups + k0 / group_size;

    const float scale = __half2float(scales[gidx]);
    const float off = SYM ? static_cast<float>(BIAS) : __half2float(zeros[gidx]);

    float v[PER_PACK];
    unpack_word<BITS, INTER>(qweight[widx], off, v);

    __half* o = out + static_cast<int64_t>(n) * K + k0;
#pragma unroll
    for (int i = 0; i < PER_PACK; ++i) o[i] = __float2half_rn(v[i] * scale);
}

// ---------------------------------------------------------------------- gemv

template <int BITS, bool SYM, bool INTER, int MT>
__global__ void __launch_bounds__(GEMV_THREADS)
gemv_kernel(const uint32_t* __restrict__ qweight,
            const __half* __restrict__ scales,
            const __half* __restrict__ zeros,
            const __half* __restrict__ x,
            __half* __restrict__ y,
            int M, int N, int K, int group_size) {
    constexpr int PER_PACK = 32 / BITS;
    constexpr int CPL = 128 / BITS;      // codes per lane per 16-byte load
    constexpr int BIAS = 1 << (BITS - 1);

    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * ROWS_PER_BLOCK + static_cast<int>(threadIdx.x >> 5);
    // Whole warps return together and there is no __syncthreads below, so this
    // early exit is safe alongside the __shfl_xor_sync reduction.
    if (n >= N) return;

    const int n_groups = K / group_size;
    const int n_chunks = K / CPL;
    const uint4* wrow = reinterpret_cast<const uint4*>(
        qweight + static_cast<int64_t>(n) * (K / PER_PACK));
    const __half* srow = scales + static_cast<int64_t>(n) * n_groups;
    const __half* zrow = SYM ? nullptr : zeros + static_cast<int64_t>(n) * n_groups;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;

    // Per-lane guard, so K need not be a multiple of 32*CPL (INT2 at K=3072).
    for (int c = lane; c < n_chunks; c += 32) {
        const uint4 w = wrow[c];
        const int k0 = c * CPL;
        const int g = k0 / group_size;   // group_size % CPL == 0, so one group
        const float scale = __half2float(srow[g]);
        const float off = SYM ? static_cast<float>(BIAS) : __half2float(zrow[g]);

        float part[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) part[m] = 0.f;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t word = j == 0 ? w.x : (j == 1 ? w.y : (j == 2 ? w.z : w.w));
            float v[PER_PACK];
            unpack_word<BITS, INTER>(word, off, v);
            const int kbase = k0 + j * PER_PACK;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (MT == 1 || m < M) {
                    part[m] += dot_x<PER_PACK>(x + static_cast<int64_t>(m) * K + kbase, v);
                }
            }
        }

#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] += scale * part[m];
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (MT == 1 || m < M) y[static_cast<int64_t>(m) * N + n] = __float2half_rn(acc[m]);
        }
    }
}

// --------------------------------------------------------------- host helpers

struct Args {
    const uint32_t* qweight;
    const __half* scales;
    const __half* zeros;
    int N;
    int K;
    int group_size;
};

Args check_weight(const torch::Tensor& qweight, const torch::Tensor& scales,
                  const std::optional<torch::Tensor>& zeros, int64_t bits,
                  int64_t group_size, bool symmetric, int64_t K) {
    TORCH_CHECK(bits == 2 || bits == 4 || bits == 8, "bits must be 2, 4 or 8, got ", bits);
    const int64_t per_pack = 32 / bits;

    TORCH_CHECK(group_size > 0, "group_size must be positive, got ", group_size);
    TORCH_CHECK(K > 0 && K % group_size == 0,
                "K=", K, " is not divisible by group_size=", group_size);

    TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be 2-D, got ", qweight.dim(), "-D");
    TORCH_CHECK(qweight.size(1) * per_pack == K,
                "qweight has ", qweight.size(1), " words per row, which is ",
                qweight.size(1) * per_pack, " codes, but K=", K);

    const int64_t N = qweight.size(0);
    const int64_t n_groups = K / group_size;

    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be float16");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == n_groups,
                "scales must be [", N, ", ", n_groups, "], got ", scales.sizes());
    TORCH_CHECK(scales.device() == qweight.device(),
                "scales and qweight must be on the same device");

    TORCH_CHECK(zeros.has_value() == !symmetric,
                symmetric ? "symmetric=true takes no zeros"
                          : "symmetric=false requires zeros");
    const __half* zptr = nullptr;
    if (zeros.has_value()) {
        const torch::Tensor& z = *zeros;
        TORCH_CHECK(z.is_cuda(), "zeros must be a CUDA tensor");
        TORCH_CHECK(z.scalar_type() == torch::kHalf, "zeros must be float16");
        TORCH_CHECK(z.is_contiguous(), "zeros must be contiguous");
        TORCH_CHECK(z.sizes() == scales.sizes(),
                    "zeros must have the same shape as scales, got ", z.sizes());
        TORCH_CHECK(z.device() == qweight.device(),
                    "zeros and qweight must be on the same device");
        zptr = reinterpret_cast<const __half*>(z.data_ptr<at::Half>());
    }

    Args a;
    a.qweight = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
    a.scales = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
    a.zeros = zptr;
    a.N = static_cast<int>(N);
    a.K = static_cast<int>(K);
    a.group_size = static_cast<int>(group_size);
    return a;
}

// bits<<2 | symmetric<<1 | interleaved
inline int variant(int64_t bits, bool symmetric, bool interleaved) {
    return static_cast<int>(bits) << 2 | static_cast<int>(symmetric) << 1
         | static_cast<int>(interleaved);
}

#define DISPATCH_VARIANT(v, BODY)                                    \
    switch (v) {                                                     \
        case (2 << 2 | 0 << 1 | 0): { BODY(2, false, false); break; } \
        case (2 << 2 | 0 << 1 | 1): { BODY(2, false, true);  break; } \
        case (2 << 2 | 1 << 1 | 0): { BODY(2, true,  false); break; } \
        case (2 << 2 | 1 << 1 | 1): { BODY(2, true,  true);  break; } \
        case (4 << 2 | 0 << 1 | 0): { BODY(4, false, false); break; } \
        case (4 << 2 | 0 << 1 | 1): { BODY(4, false, true);  break; } \
        case (4 << 2 | 1 << 1 | 0): { BODY(4, true,  false); break; } \
        case (4 << 2 | 1 << 1 | 1): { BODY(4, true,  true);  break; } \
        case (8 << 2 | 0 << 1 | 0): { BODY(8, false, false); break; } \
        case (8 << 2 | 0 << 1 | 1): { BODY(8, false, true);  break; } \
        case (8 << 2 | 1 << 1 | 0): { BODY(8, true,  false); break; } \
        case (8 << 2 | 1 << 1 | 1): { BODY(8, true,  true);  break; } \
        default: TORCH_CHECK(false, "unsupported variant ", v);        \
    }

}  // namespace

torch::Tensor dequantize(torch::Tensor qweight, torch::Tensor scales,
                         std::optional<torch::Tensor> zeros, int64_t bits, int64_t group_size,
                         bool symmetric, bool interleaved, int64_t K) {
    const Args a = check_weight(qweight, scales, zeros, bits, group_size, symmetric, K);
    const int64_t per_pack = 32 / bits;
    TORCH_CHECK(group_size % per_pack == 0,
                "group_size=", group_size, " must be a multiple of ", per_pack,
                " values per uint32 word at ", bits, " bits");

    const c10::cuda::CUDAGuard guard(qweight.device());
    auto out = torch::empty({static_cast<int64_t>(a.N), K}, scales.options());
    __half* outp = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    const int64_t total = qweight.numel();
    const int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_DEQ(B, S, I)                                                         \
    dequantize_kernel<B, S, I><<<blocks, threads, 0, stream>>>(                     \
        a.qweight, a.scales, a.zeros, outp, total, a.N, a.K, a.group_size)

    DISPATCH_VARIANT(variant(bits, symmetric, interleaved), LAUNCH_DEQ);
#undef LAUNCH_DEQ

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
                   std::optional<torch::Tensor> zeros, int64_t bits, int64_t group_size,
                   bool symmetric, bool interleaved) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [M, K], got ", x.dim(), "-D");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    TORCH_CHECK(M >= 1 && M <= MAX_M, "gemv handles 1 <= M <= ", MAX_M,
                ", got M=", M, "; use the dequantize + F.linear fallback");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x must be 16-byte aligned");

    const Args a = check_weight(qweight, scales, zeros, bits, group_size, symmetric, K);
    TORCH_CHECK(x.device() == qweight.device(), "x and qweight must be on the same device");

    const int64_t cpl = 128 / bits;
    TORCH_CHECK(group_size % cpl == 0,
                "group_size=", group_size, " must be a multiple of ", cpl,
                " (the codes one lane loads per uint4) at ", bits, " bits; this "
                "excludes INT2 with group_size 32");

    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({M, static_cast<int64_t>(a.N)}, x.options());
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    const int blocks = (a.N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = static_cast<int>(M);

#define LAUNCH_GEMV(B, S, I)                                                        \
    if (m == 1) {                                                                   \
        gemv_kernel<B, S, I, 1><<<blocks, GEMV_THREADS, 0, stream>>>(               \
            a.qweight, a.scales, a.zeros, xp, yp, m, a.N, a.K, a.group_size);       \
    } else {                                                                        \
        gemv_kernel<B, S, I, MAX_M><<<blocks, GEMV_THREADS, 0, stream>>>(           \
            a.qweight, a.scales, a.zeros, xp, yp, m, a.N, a.K, a.group_size);       \
    }

    DISPATCH_VARIANT(variant(bits, symmetric, interleaved), LAUNCH_GEMV);
#undef LAUNCH_GEMV

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
