#pragma once

// Batched scalar-contraction (dot product) kernel for the small tier.
//
// Shapes whose free dims squash to M = N = 1 contract every mode: each desc
// is a length-K dot product of two dense blocks. The warp GEMM maps such a
// desc to a single warp with ONE active lane serializing K dependent loads —
// pathological for large K (DMRG Lanczos orthogonalization dots reach
// K ~ 40k at ~0.1 GFLOPS). This kernel assigns one CTA per desc: threads
// stride K, reduce via shuffle + shared memory, one thread writes the scalar.
//
// Shares the WarpShape/WarpMem arrays with the warp kernel: descs routed
// here get zero-width entries in the warp prefix, so the two kernels
// partition the small-tier desc list without duplicated staging. Summation
// order is free (addition commutes) — the K strides encode the A<->B
// element pairing, which collapses to a flat unit-stride loop whenever the
// squashed layouts match (the common case; squash_dimensions_dual already
// merges matched contiguous dims).

#include "../../warp_desc.hpp"

#include <cuda_runtime.h>
#include <type_traits>

namespace bsptc {

constexpr int DOT_REDUCE_BLOCK = 256;

template <typename T, bool INT_LARGE, bool INT_K>
__global__ void dot_reduce_kernel(
    const WarpShape<
        std::conditional_t<INT_LARGE, IntDim, StdDim>,
        std::conditional_t<INT_K, IntK, StdK>
    >* __restrict__ shapes,
    const WarpMem* __restrict__ mems,
    const int32_t* __restrict__ dot_indices,
    int num_dots)
{
    const int di = static_cast<int>(blockIdx.x);
    if (di >= num_dots) return;

    const WarpMem& mem = mems[dot_indices[di]];
    const auto& shape = shapes[mem.shape_idx];

    const T* __restrict__ A = static_cast<const T*>(mem.A);
    const T* __restrict__ B = static_cast<const T*>(mem.B);

    const int tid = static_cast<int>(threadIdx.x);
    const int nt = static_cast<int>(blockDim.x);
    const int32_t K = shape.K_total;

    T sum = T(0);

    if constexpr (!INT_K) {
        const dim_t sa = shape.k.stride_a;
        const dim_t sb = shape.k.stride_b;
        if (sa == 1 && sb == 1) {
            for (int32_t k = tid; k < K; k += nt)
                sum += A[k] * B[k];
        } else {
            for (int32_t k = tid; k < K; k += nt)
                sum += A[static_cast<int64_t>(k) * sa] *
                       B[static_cast<int64_t>(k) * sb];
        }
    } else {
        const int nd = shape.k.num_dims;
        if (nd == 1) {
            const dim_t sa = shape.k.strides_a[0];
            const dim_t sb = shape.k.strides_b[0];
            if (sa == 1 && sb == 1) {
                for (int32_t k = tid; k < K; k += nt)
                    sum += A[k] * B[k];
            } else {
                for (int32_t k = tid; k < K; k += nt)
                    sum += A[static_cast<int64_t>(k) * sa] *
                           B[static_cast<int64_t>(k) * sb];
            }
        } else {
            for (int32_t k = tid; k < K; k += nt) {
                int32_t t = k;
                int64_t oa = 0, ob = 0;
                #pragma unroll
                for (int i = WARP_MAX_DIMS - 1; i >= 0; --i) {
                    if (i >= nd) continue;
                    const int32_t c = t % shape.k.dims[i];
                    t /= shape.k.dims[i];
                    oa += static_cast<int64_t>(c) * shape.k.strides_a[i];
                    ob += static_cast<int64_t>(c) * shape.k.strides_b[i];
                }
                sum += A[oa] * B[ob];
            }
        }
    }

    // Block reduce: intra-warp shuffle, then cross-warp via shared memory.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ T warp_sums[DOT_REDUCE_BLOCK / 32];
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        const int nwarps = nt >> 5;
        sum = (lane < nwarps) ? warp_sums[lane] : T(0);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
        if (lane == 0) {
            T* C = static_cast<T*>(mem.C);
            if (mem.use_atomic)
                atomicAdd(C, sum);
            else
                *C += sum;
        }
    }
}

// variant carries the same 4-bit encoding as launch_warp_gemm; only the
// INT_LARGE / INT_K bits matter here (they select the WarpShape struct
// layout) — contiguity is detected per shape at runtime.
template <typename T>
void launch_dot_reduce(const void* d_shapes, const WarpMem* d_mems,
                       const int32_t* d_dot_indices, int num_dots,
                       int variant, cudaStream_t stream);

} // namespace bsptc
