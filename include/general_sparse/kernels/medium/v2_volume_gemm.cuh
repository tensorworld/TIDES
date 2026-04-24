#pragma once

#include "../transpose/nd_hypertile.cuh"
#include "../../device_props.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>

// Volume kernel: loads entire A (M×K) and B (K×N) into shared memory via
// scatter-gather, then computes C with a fixed 2×2 register tile per thread.
//
// Condition: ceil(M/2)*ceil(N/2) ≤ CTA_SIZE AND M*K + K*N ≤ vol_threshold.
// No K tiling — the full K dimension fits in SMEM.
//
// Persistent dispatch: each CTA loops over problems (stride = gridDim.x),
// amortizing launch overhead when there are many small problems.
//
// Two instantiations:
//   volume_kernel<16, 16>  →  64 threads  (tiles ≤ 64,  e.g. M=8  N=16)
//   volume_kernel<32, 32>  → 256 threads  (tiles ≤ 256, e.g. M=6  N=64)
//
// Thread-to-tile mapping is linear (tid → 2×2 output tile) so that idle
// threads pack into the highest-numbered warps, minimizing divergence.
namespace bsptc {

static constexpr int VOL_MAX_RANK = 5;
// VOL_THRESHOLD defined in work_partitioner.hpp (canonical source).
// M,N ≤ 512 (uint16), K ≤ 4096 (uint16), volumes ≤ 8192 (uint16),
// strides ≤ 8192 (uint16).  Post-squash rank guard in setup_medium_volume_descs
// rejects problems whose combined (free + K) squashed rank exceeds 5.

// ============================================================================
// Shape/mem split: shared geometry (VolumeShape, few unique per contraction)
// + lightweight per-instance data (VolumeMem, 32 bytes).
// ============================================================================

// Packed geometry — fields narrowed to minimum widths.
// Eligibility: ceil(M/2)*ceil(N/2) ≤ 256 AND M*K+N*K ≤ vol_threshold.
// Layout: 4-byte-aligned divmods first, then 2-byte, then 1-byte fields.
struct VolumeShape {
    GridDivmod A_divmod[VOL_MAX_RANK]; // 60
    GridDivmod B_divmod[VOL_MAX_RANK]; // 60
    GridDivmod K_divmod;               // 12
    uint16_t A_stride[VOL_MAX_RANK];   // 10 (max stride ≤ vol_threshold = 8192)
    uint16_t B_stride[VOL_MAX_RANK];   // 10
    uint16_t A_volume;                 // 2  (M*K ≤ 8192)
    uint16_t B_volume;                 // 2  (N*K ≤ 8192)
    uint16_t K;                        // 2  (≤ 4096)
    uint16_t M;                        // 2  (≤ 512 in practice)
    uint16_t N;                        // 2  (≤ 512 in practice)
    int8_t   A_rank;                   // 1
    int8_t   B_rank;                   // 1
    // sizeof = 166, compiler adds 2 bytes tail padding → 168, divisible by 4
};

struct VolumeMem {
    const void* A;
    const void* B;
    void* C;
    uint32_t shape_idx;
    int8_t use_atomic;
    int8_t flush_output;
};

static_assert(std::is_trivially_copyable_v<VolumeShape>);
static_assert(std::is_trivially_copyable_v<VolumeMem>);

// ============================================================================
// Device helper: compute scattered global offset from flat linear index.
// ============================================================================

template <int R, typename StrideT>
__device__ __forceinline__
uint32_t vol_offset(int idx, int rank,
                    const GridDivmod divmod[],
                    const StrideT stride[]) {
    uint32_t urem = static_cast<uint32_t>(idx);
    uint32_t off = 0;
    #pragma unroll
    for (int i = 0; i < R - 1; ++i) {
        if (i < rank - 1) {
            uint32_t q, r;
            divmod[i].divmod(urem, q, r);
            urem = q;
            off += r * static_cast<uint32_t>(stride[i]);
        }
    }
    off += urem * static_cast<uint32_t>(stride[rank - 1]);
    return off;
}

// ============================================================================
// Volume GEMM kernel
// ============================================================================

template <typename T, int MaxM, int MaxN>
__global__ void volume_gemm_kernel(const VolumeShape* __restrict__ shapes,
                                   const VolumeMem* __restrict__ mems,
                                   int32_t num_problems) {
    constexpr int R = VOL_MAX_RANK;
    constexpr int BLOCK_SIZE = (MaxM / 2) * (MaxN / 2);
    const int tid = threadIdx.x;

    __shared__ VolumeShape s_shape;
    __shared__ VolumeMem s_mem;
    extern __shared__ char smem_raw[];

    // Contiguous block dispatch: each CTA gets a consecutive range of problems
    // so that same-output problems (sorted adjacent) can fuse in registers.
    const int probs_per_cta = (num_problems + gridDim.x - 1) / gridDim.x;
    const int start = blockIdx.x * probs_per_cta;
    const int end = min(start + probs_per_cta, num_problems);

    uint32_t cur_shape_idx = 0xFFFFFFFF;

    int prob = start;
    while (prob < end) {
        // ---- Start of output run ----
        // Load mem (28 bytes — single thread)
        if (tid == 0) s_mem = mems[prob];
        __syncthreads();

        // Load shape cooperatively (only if shape changed)
        if (s_mem.shape_idx != cur_shape_idx) {
            cur_shape_idx = s_mem.shape_idx;
            constexpr int WORDS = (sizeof(VolumeShape) + sizeof(uint32_t) - 1)
                                  / sizeof(uint32_t);
            auto* dst_w = reinterpret_cast<uint32_t*>(&s_shape);
            const auto* src_w = reinterpret_cast<const uint32_t*>(&shapes[cur_shape_idx]);
            for (int i = tid; i < WORDS; i += BLOCK_SIZE)
                dst_w[i] = src_w[i];
        }
        __syncthreads();

        // Output-constant fields for this run (same M,N,C for all fused problems)
        T* __restrict__ C = static_cast<T*>(s_mem.C);
        const int M = s_shape.M, N = s_shape.N;
        const int8_t is_atomic = s_mem.use_atomic;

        // 2×2 tile mapping (constant within run: same M,N)
        const int tiles_n = (N + 1) / 2;
        const int num_tiles = ((M + 1) / 2) * tiles_n;
        const int tm = tid / tiles_n;
        const int tn = tid % tiles_n;
        const int m0 = tm * 2;
        const int n0 = tn * 2;
        const bool has_m1 = (m0 + 1 < M);
        const bool has_n1 = (n0 + 1 < N);

        T c00 = T(0), c01 = T(0), c10 = T(0), c11 = T(0);

        // ---- Accumulate loop: fuse consecutive same-C problems ----
        for (;;) {
            const int K = s_shape.K;
            const int stride = K + 1;
            T* smem = reinterpret_cast<T*>(smem_raw);
            T* sA = smem;
            T* sB = smem + M * stride;

            // Load A (M×K) from strided global to dense SMEM
            const T* __restrict__ A = static_cast<const T*>(s_mem.A);
            for (int i = tid; i < s_shape.A_volume; i += BLOCK_SIZE) {
                uint32_t m_u, k_u;
                s_shape.K_divmod.divmod(static_cast<uint32_t>(i), m_u, k_u);
                uint32_t off = vol_offset<R>(i, s_shape.A_rank, s_shape.A_divmod, s_shape.A_stride);
                sA[m_u * stride + k_u] = A[off];
            }

            // Load B (N×K) from strided global to dense SMEM
            const T* __restrict__ B = static_cast<const T*>(s_mem.B);
            for (int i = tid; i < s_shape.B_volume; i += BLOCK_SIZE) {
                uint32_t n_u, k_u;
                s_shape.K_divmod.divmod(static_cast<uint32_t>(i), n_u, k_u);
                uint32_t off = vol_offset<R>(i, s_shape.B_rank, s_shape.B_divmod, s_shape.B_stride);
                sB[n_u * stride + k_u] = B[off];
            }

            __syncthreads();

            // 2×2 register-tile compute — accumulate into run registers
            if (tid < num_tiles) {
                const T* sA_row0 = sA + m0 * stride;
                const T* sA_row1 = sA + (m0 + 1) * stride;
                const T* sB_row0 = sB + n0 * stride;
                const T* sB_row1 = sB + (n0 + 1) * stride;

                for (int k = 0; k < K; ++k) {
                    T a0 = sA_row0[k];
                    T b0 = sB_row0[k];
                    c00 += a0 * b0;
                    if (has_n1) c01 += a0 * sB_row1[k];
                    if (has_m1) {
                        T a1 = sA_row1[k];
                        c10 += a1 * b0;
                        if (has_n1) c11 += a1 * sB_row1[k];
                    }
                }
            }

            ++prob;

            // End of run: flush if last-in-group or at CTA boundary
            if (s_mem.flush_output || prob >= end) break;

            // Same output continues — load next mem + shape if needed
            __syncthreads();
            if (tid == 0) s_mem = mems[prob];
            __syncthreads();

            if (s_mem.shape_idx != cur_shape_idx) {
                cur_shape_idx = s_mem.shape_idx;
                constexpr int WORDS = (sizeof(VolumeShape) + sizeof(uint32_t) - 1)
                                      / sizeof(uint32_t);
                auto* dst_w = reinterpret_cast<uint32_t*>(&s_shape);
                const auto* src_w = reinterpret_cast<const uint32_t*>(&shapes[cur_shape_idx]);
                for (int i = tid; i < WORDS; i += BLOCK_SIZE)
                    dst_w[i] = src_w[i];
                __syncthreads();
            }
        }

        // ---- Flush: write accumulated result to global ----
        if (tid < num_tiles) {
            if (is_atomic) {
                atomicAdd(&C[m0 * N + n0], c00);
                if (has_n1) atomicAdd(&C[m0 * N + n0 + 1], c01);
                if (has_m1) {
                    atomicAdd(&C[(m0 + 1) * N + n0], c10);
                    if (has_n1) atomicAdd(&C[(m0 + 1) * N + n0 + 1], c11);
                }
            } else {
                C[m0 * N + n0] += c00;
                if (has_n1) C[m0 * N + n0 + 1] += c01;
                if (has_m1) {
                    C[(m0 + 1) * N + n0] += c10;
                    if (has_n1) C[(m0 + 1) * N + n0 + 1] += c11;
                }
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// Host launch helpers
// ============================================================================

// SMEM guard: +1 row for sB so that sB_row1 reads are always in-bounds
// when N is odd and the last tile reads one row past sB.
inline int volume_smem_elems(int M, int N, int K) {
    int stride = K + 1;
    return (M + N + 1) * stride;
}

template <typename T, int MaxM, int MaxN>
void launch_volume_gemm(
    const VolumeShape* d_shapes,
    const VolumeMem* d_mems,
    int num_problems,
    int max_smem_elems,
    cudaStream_t stream)
{
    if (num_problems == 0) return;

    constexpr int BLOCK_SIZE = (MaxM / 2) * (MaxN / 2);
    auto kern = volume_gemm_kernel<T, MaxM, MaxN>;
    int smem_bytes = max_smem_elems * static_cast<int>(sizeof(T));

    int grid_size = persistent_grid_size(kern, BLOCK_SIZE, smem_bytes, num_problems);
    kern<<<grid_size, BLOCK_SIZE, smem_bytes, stream>>>(d_shapes, d_mems, num_problems);
}

// Dispatch helper: select kernel instantiation based on max tile count.
template <typename T>
void launch_volume_gemm_dispatch(
    const VolumeShape* d_shapes,
    const VolumeMem* d_mems,
    int num_problems,
    int max_smem_elems,
    int max_tiles,   // max ceil(M/2)*ceil(N/2) across all problems
    cudaStream_t stream)
{
    if (max_tiles <= 64) {
        launch_volume_gemm<T, 16, 16>(d_shapes, d_mems, num_problems, max_smem_elems, stream);
    } else {
        launch_volume_gemm<T, 32, 32>(d_shapes, d_mems, num_problems, max_smem_elems, stream);
    }
}

} // namespace bsptc
