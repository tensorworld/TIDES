#pragma once

#include "../../warp_desc.hpp"
#include <cuda_runtime.h>
#include <type_traits>

namespace bsptc {

// ============================================================================
// Warp-level small GEMM kernel — canonical small/large form.
//
// Template params: INT_LARGE (large free dim is IntDim), INT_K (K is IntK),
// CONTIG_K_A/B (innermost K stride is 1 in A/B).
//
// The smaller free dim (≤ 3) is always stored as a bare stride in the shape.
// A runtime flag (m_is_small) routes the small/large offsets to the correct
// operand (A or B) and computes the correct output index.
// ============================================================================

template <typename T, bool INT_LARGE, bool INT_K,
          bool CONTIG_K_A, bool CONTIG_K_B>
__global__ void warp_gemm_kernel(
    const WarpShape<
        std::conditional_t<INT_LARGE, IntDim, StdDim>,
        std::conditional_t<INT_K, IntK,   StdK>
    >* __restrict__ shapes,
    const WarpMem* __restrict__ mems,
    const int32_t* __restrict__ warp_prefix,
    int num_descs,
    int total_warps)
{
    using ShapeT = WarpShape<
        std::conditional_t<INT_LARGE, IntDim, StdDim>,
        std::conditional_t<INT_K, IntK,   StdK>>;

    const int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int grid_warps = (gridDim.x * blockDim.x) >> 5;
    const int lane_id = threadIdx.x & 31;

    const int items_per_warp = (total_warps + grid_warps - 1) / grid_warps;
    const int warp_begin = global_warp_id * items_per_warp;
    const int warp_end = min(warp_begin + items_per_warp, total_warps);

    int desc_hint = 0;
    if (warp_begin < total_warps) {
        int lo = 0, hi = num_descs - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (warp_prefix[mid + 1] <= warp_begin) lo = mid + 1;
            else hi = mid;
        }
        desc_hint = lo;
    }

    int prev_desc = -1;
    uint32_t prev_shape_idx = 0xFFFFFFFF;
    const ShapeT* shape_ptr = nullptr;
    const WarpMem* mem_ptr = nullptr;
    int32_t cached_small = 0, cached_large = 0;

    for (int warp_idx = warp_begin; warp_idx < warp_end; ++warp_idx) {
        while (desc_hint < num_descs - 1 && warp_prefix[desc_hint + 1] <= warp_idx)
            ++desc_hint;

        const bool desc_changed = (desc_hint != prev_desc);

        if (desc_changed) {
            prev_desc = desc_hint;
            mem_ptr = &mems[desc_hint];
            const uint32_t new_shape_idx = mem_ptr->shape_idx;
            if (new_shape_idx != prev_shape_idx) {
                prev_shape_idx = new_shape_idx;
                shape_ptr = &shapes[new_shape_idx];
            }
        }

        const auto& shape = *shape_ptr;
        const auto& mem = *mem_ptr;

        const int32_t S = shape.small_size;
        const int32_t L = shape.large_size;
        const int32_t SL = S * L;
        const int32_t element_idx = (warp_idx - warp_prefix[desc_hint]) * 32 + lane_id;

        if (element_idx >= SL) continue;

        // Decompose element_idx = si * L + li
        int32_t si, li;
        if (desc_changed) {
            if (S == 1) {
                si = 0; li = element_idx;
            } else if (S == 2) {
                si = (element_idx >= L) ? 1 : 0;
                li = element_idx - si * L;
            } else if (S == 3) {
                const int32_t twoL = L + L;
                if (element_idx >= twoL) { si = 2; li = element_idx - twoL; }
                else if (element_idx >= L) { si = 1; li = element_idx - L; }
                else { si = 0; li = element_idx; }
            } else {
                si = element_idx / L;
                li = element_idx - si * L;
            }
        } else {
            si = cached_small;
            li = cached_large + 32;
            if (S >= 2 && li >= L) { li -= L; si++; }
            if (S >= 3 && li >= L) { li -= L; si++; }
        }
        cached_small = si;
        cached_large = li;

        // --- Decode large dim offset ---
        int32_t large_off = 0;
        if constexpr (INT_LARGE) {
            int32_t temp = li;
            for (int i = shape.large.num_dims - 1; i >= 0; --i) {
                large_off += (temp % shape.large.dims[i]) * shape.large.strides[i];
                temp /= shape.large.dims[i];
            }
        } else {
            large_off = li * shape.large.stride;
        }

        // --- Small dim offset (always a single stride) ---
        const int32_t small_off = si * shape.small_stride;

        // --- Route to A/B based on which dim is small ---
        int32_t offset_A, offset_B;
        if (shape.m_is_small) {
            offset_A = small_off;   // A owns the small free dim (M)
            offset_B = large_off;   // B owns the large free dim (N)
        } else {
            offset_A = large_off;   // A owns the large free dim (M)
            offset_B = small_off;   // B owns the small free dim (N)
        }

        const T* A_ptr = static_cast<const T*>(mem.A);
        const T* B_ptr = static_cast<const T*>(mem.B);

        T sum = T(0);

        if constexpr (!INT_K) {
            const dim_t stride_A_k = shape.k.stride_a;
            const dim_t stride_B_k = shape.k.stride_b;
            const int32_t K_total = shape.K_total;

            for (int32_t k = 0; k < K_total; ++k) {
                int32_t addr_A = offset_A + (CONTIG_K_A ? k : k * stride_A_k);
                int32_t addr_B = offset_B + (CONTIG_K_B ? k : k * stride_B_k);
                sum += A_ptr[addr_A] * B_ptr[addr_B];
            }
        } else {
            const int8_t num_k_dims = shape.k.num_dims;
            const dim_t inner_k_dim = shape.k.dims[num_k_dims - 1];
            const dim_t outer_k_iters = shape.K_total / inner_k_dim;
            const dim_t stride_A_k_inner = shape.k.strides_a[num_k_dims - 1];
            const dim_t stride_B_k_inner = shape.k.strides_b[num_k_dims - 1];

            // Hoist outer-dim fields — compile-time indexed in unrolled
            // odometer, so the compiler keeps them in registers.
            dim_t l_k_dims[WARP_MAX_DIMS - 1];
            dim_t l_k_strides_a[WARP_MAX_DIMS - 1];
            dim_t l_k_strides_b[WARP_MAX_DIMS - 1];
            #pragma unroll
            for (int i = 0; i < WARP_MAX_DIMS - 1; ++i) {
                l_k_dims[i] = shape.k.dims[i];
                l_k_strides_a[i] = shape.k.strides_a[i];
                l_k_strides_b[i] = shape.k.strides_b[i];
            }

            int32_t k_coord[WARP_MAX_DIMS - 1] = {0};
            int32_t offset_A_k = 0, offset_B_k = 0;

            for (int32_t k_outer = 0; k_outer < outer_k_iters; ++k_outer) {
                for (int32_t k = 0; k < inner_k_dim; ++k) {
                    int32_t a_k = offset_A_k + (CONTIG_K_A ? k : k * stride_A_k_inner);
                    int32_t b_k = offset_B_k + (CONTIG_K_B ? k : k * stride_B_k_inner);
                    sum += A_ptr[offset_A + a_k] * B_ptr[offset_B + b_k];
                }

                #pragma unroll
                for (int c = WARP_MAX_DIMS - 2; c >= 0; --c) {
                    if (c >= num_k_dims - 1) continue;
                    k_coord[c]++;
                    if (k_coord[c] < l_k_dims[c]) {
                        offset_A_k += l_k_strides_a[c];
                        offset_B_k += l_k_strides_b[c];
                        break;
                    } else {
                        k_coord[c] = 0;
                        offset_A_k -= l_k_strides_a[c] * (l_k_dims[c] - 1);
                        offset_B_k -= l_k_strides_b[c] * (l_k_dims[c] - 1);
                    }
                }
            }
        }

        // Output: C is always [M, N] row-major.
        T* C_ptr = static_cast<T*>(mem.C);
        int32_t c_idx;
        if (shape.m_is_small) {
            // M=small, N=large → c_idx = m*N + n = si*L + li = element_idx
            c_idx = element_idx;
        } else {
            // M=large, N=small → c_idx = m*N + n = li*S + si
            c_idx = li * S + si;
        }
        if (mem.use_atomic)
            atomicAdd(&C_ptr[c_idx], sum);
        else
            C_ptr[c_idx] += sum;
    }
}

template <typename T>
void launch_warp_gemm(const void* d_shapes, const WarpMem* d_mems,
                      const int32_t* d_warp_prefix, int num_descs,
                      int total_warps, int variant, cudaStream_t stream);

} // namespace bsptc
