#pragma once

// Lightweight header for warp GEMM types — does NOT pull in the heavy
// preprocessing_kernel.cuh include chain.  Used by warp_gemm.cuh and
// warp_gemm_instantiate.cu.
//
// Canonical small/large form: for small GEMM (min(M,N) < 4), the smaller
// free dim is always ≤ 3 and effectively 1D, stored as a bare stride.
// Only the larger free dim may need multi-dim (IntDim) decode.

#include <types.hpp>        // dim_t
#include <cstdint>

namespace bsptc {

constexpr int WARP_MAX_DIMS = 4;

// --- Building blocks for the large free dim ---
struct StdDim { dim_t stride; };
struct IntDim {
    int8_t num_dims;
    dim_t dims[WARP_MAX_DIMS];
    dim_t strides[WARP_MAX_DIMS];
};

// --- Building blocks for contracted dims (K) ---
struct StdK { dim_t stride_a; dim_t stride_b; };
struct IntK {
    int8_t num_dims;
    dim_t dims[WARP_MAX_DIMS];
    dim_t strides_a[WARP_MAX_DIMS];
    dim_t strides_b[WARP_MAX_DIMS];
    dim_t k_start;   // flat K offset for K-split (0 if no split)
};

// Shape-only part of a warp descriptor (shared across descs with same geometry).
// Canonical form: the smaller free dim (size ≤ 3) is stored as a bare stride;
// the larger free dim uses StdDim or IntDim via the LargeFree template param.
// m_is_small: 1 if M (A's free) is the small dim, 0 if N (B's free) is small.
template <typename LargeFree, typename Contracted>
struct WarpShape {
    dim_t small_size;       // 1, 2, or 3 (may be larger in synthetic benchmarks)
    dim_t large_size;       // up to 1024
    dim_t K_total;
    dim_t small_stride;     // stride of small dim in its source operand
    LargeFree large;        // decode info for large dim in its source operand
    Contracted k;
    int8_t m_is_small;      // 1 → M is the small dim; 0 → N is the small dim
};

// Memory-only part of a warp descriptor (unique per desc).
struct WarpMem {
    const void* A;
    const void* B;
    void* C;
    uint32_t shape_idx;
    int8_t use_atomic;
};

// Traits-only descriptor: maps (LargeFree, Contracted) to type aliases.
// Used as template parameter for prepare_small_shapes_impl dispatch.
template <typename LargeFree, typename Contracted>
struct WarpDesc {
    using LargeFreeType = LargeFree;
    using ContractedType = Contracted;
};

using DescSS = WarpDesc<StdDim, StdK>;
using DescIS = WarpDesc<IntDim, StdK>;
using DescSI = WarpDesc<StdDim, IntK>;
using DescII = WarpDesc<IntDim, IntK>;

using ShapeSS = WarpShape<StdDim, StdK>;

} // namespace bsptc
