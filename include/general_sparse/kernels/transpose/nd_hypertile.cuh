#pragma once

// GCC 13+ emits false -Warray-bounds warnings when std::sort operates on
// stack arrays (int32_t[HT_MAX_RANK]) indexed by a runtime variable.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cuda_runtime.h>

#include "../../device_props.hpp"
#include "../../block_shape_map.hpp"
#include "../../../layout_utils.hpp"

// Hypertile ND Transpose: zero-ALU hot-loop transpose kernel.
// NOTE: This header must be included AFTER gpu_sparse_device.cuh,
// gpu_sparse_input.hpp, and gemm_layout_check.hpp (for GPUSparseInput,
// GemmLayoutPlan, check_cuda, dim_t, etc.).
//
// Key ideas:
// 1. Dimension squashing: merge adjacent dims with same relative order in/out
// 2. Per-tensor register map: precompute read/write offsets + SMEM routing once
//    per tensor, reuse across all tiles of that tensor
// 3. Stateful CTA dispatch: binary search only on tensor boundary crossing
// 4. Flat SMEM buffer with scattered read (no 2D tile constraint)

namespace bsptc {

// Global tiler selection for A/B benchmarking.  Default = greedy (false).
// Set to true before calling build_task_for_shape / build_hypertile_index
// to use the exhaustive-search tiler instead.
inline bool g_use_search_tiler = false;

constexpr int HT_MAX_RANK = 8;

// ---------------------------------------------------------------------------
// TileDivmod: branchless 32-bit divmod for n ≤ 1024 (tile-local indices).
// Uses fixed shift of 21: magic = (1<<21)/d + 1. Max error = n/2^21 < 1/2048,
// which is below the minimum fractional gap 1/d ≥ 1/1024. Exact for all n ≤ 1024.
// 3 instructions: IMUL + SHR + IMAD. No __umulhi, no branch. 8 bytes.
// ---------------------------------------------------------------------------
struct TileDivmod {
    uint32_t magic;    // (1u << 21) / divisor + 1
    uint32_t divisor;

    __device__ __forceinline__
    void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
        q = (n * magic) >> 21;
        r = n - q * divisor;
    }
};

inline TileDivmod make_tile_divmod(uint32_t d) {
    TileDivmod td;
    td.divisor = d;
    td.magic = (1u << 21) / d + 1;
    return td;
}

// ---------------------------------------------------------------------------
// GridDivmod: branchless __umulhi divmod for the odometer (n ≤ ~4M).
// GR packing guarantees divisor > 1 for transpose grids, but the GEMM kernel
// uses this for K_tile_divmod which can be 1.  The d<=1 branch is warp-uniform
// and free on GPU.  12 bytes.
// ---------------------------------------------------------------------------
struct GridDivmod {
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift_right;

    __device__ __forceinline__
    void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
        if (divisor <= 1) { q = n; r = 0; return; }
        q = __umulhi(n, multiplier) >> shift_right;
        r = n - q * divisor;
    }
};

inline GridDivmod make_grid_divmod(uint32_t d) {
    GridDivmod gd;
    gd.divisor = d;
    if (d <= 1) { gd.multiplier = 0; gd.shift_right = 0; return gd; }
    uint32_t log2_d = 31 - __builtin_clz(d);
    if ((1U << log2_d) < d) log2_d++;
    uint32_t p = 31 + log2_d;
    gd.multiplier = static_cast<uint32_t>(((1ULL << p) + d - 1) / d);
    gd.shift_right = p - 32;
    return gd;
}

// ---------------------------------------------------------------------------
// Reusable sub-structs for read/write paths.
// These are embedded in HypertileShape and can also be embedded in other
// task types (e.g., HypertileGemmTask) for the fused GEMM loader.
// ---------------------------------------------------------------------------
// Field order: uint32_t first, then TileDivmod (uint32_t pair), then uint16_t,
// then uint8_t — all 4-byte or smaller, no padding needed.
template <int R>
struct HypertileReadInfo {
    uint32_t   src_stride[R];     // global source strides (element-level, < 2^32)
    TileDivmod divmod[R];         // tile decomposition (RR entries, raster-squashed)
    uint16_t   edge_allowed[R];   // precomputed edge bound (0xFFFF = no edge)
    uint8_t    grid_idx[R];       // maps RR idx -> GR idx (0xFF = fulfilled)
    uint8_t    RR;                // read-squashed rank
};

template <int R>
struct HypertileWriteInfo {
    uint32_t   dst_stride[R];     // global dest strides (element-level, < 2^32)
    TileDivmod divmod[R];         // tile decomposition (WR entries, raster-squashed)
    uint16_t   smem_stride[R];    // SMEM routing strides
    uint16_t   edge_allowed[R];   // precomputed edge bound (0xFFFF = no edge)
    uint8_t    grid_idx[R];       // maps WR idx -> GR idx (0xFF = fulfilled)
};

// ---------------------------------------------------------------------------
// HypertileShape: per-shape tiling metadata (shared across blocks with same
// dimensions under the permutation). Combined with HypertileMem (per-block
// unique pointers) for GPU upload: N×sizeof(Mem) + K×sizeof(Shape) where K << N.
// ---------------------------------------------------------------------------
template <int R = HT_MAX_RANK>
struct HypertileShape {
    uint32_t num_tiles;
    uint16_t tile_volume;
    uint8_t squashed_rank, GR, WR;
    uint8_t _pad[3];
    GridDivmod grid_divmod[R];
    uint32_t grid_src_step[R], grid_dst_step[R];
    uint16_t edge_sizes[R];
    HypertileReadInfo<R> read;
    HypertileWriteInfo<R> write;
};

struct HypertileMem {
    int64_t src_base;
    int64_t dst_base;
    uint32_t shape_idx;
    uint8_t _pad[4];  // align to 24 bytes
};

// Convenience alias for host-side code (max-rank shapes).
using HypertileShapeMax = HypertileShape<HT_MAX_RANK>;

// Pack a max-rank shape into a smaller HypertileShape<R>.
template <int R>
HypertileShape<R> pack_shape(const HypertileShapeMax& src) {
    HypertileShape<R> dst{};
    dst.num_tiles = src.num_tiles;
    dst.tile_volume = src.tile_volume;
    dst.squashed_rank = src.squashed_rank;
    dst.GR = src.GR;
    dst.WR = src.WR;
    for (int i = 0; i < R; ++i) {
        dst.grid_divmod[i]    = src.grid_divmod[i];
        dst.grid_src_step[i]  = src.grid_src_step[i];
        dst.grid_dst_step[i]  = src.grid_dst_step[i];
        dst.edge_sizes[i]     = src.edge_sizes[i];
        dst.read.divmod[i]    = src.read.divmod[i];
        dst.read.src_stride[i]= src.read.src_stride[i];
        dst.read.edge_allowed[i] = src.read.edge_allowed[i];
        dst.read.grid_idx[i]  = src.read.grid_idx[i];
        dst.write.divmod[i]    = src.write.divmod[i];
        dst.write.dst_stride[i]= src.write.dst_stride[i];
        dst.write.smem_stride[i]= src.write.smem_stride[i];
        dst.write.edge_allowed[i] = src.write.edge_allowed[i];
        dst.write.grid_idx[i]  = src.write.grid_idx[i];
    }
    dst.read.RR = src.read.RR;
    return dst;
}

// Host-only diagnostic info per task (not sent to GPU).
struct HypertileTaskInfo {
    int32_t squashed_rank;
    int32_t tile_shape[HT_MAX_RANK];    // dim-order
    int32_t full_shape[HT_MAX_RANK];    // dim-order
    int32_t grid_shape[HT_MAX_RANK];    // dim-order
    int32_t read_order[HT_MAX_RANK];
    int32_t write_order[HT_MAX_RANK];
    int64_t src_stride[HT_MAX_RANK];    // dim-order
    int64_t dst_stride[HT_MAX_RANK];    // dim-order
};

// ============================================================================
// Host-side preprocessing
// ============================================================================

// Result of dimension squashing.
struct SquashResult {
    int32_t rank;                        // squashed rank
    int32_t shape[HT_MAX_RANK];          // squashed shape (one entry per squashed dim)
    int64_t src_stride[HT_MAX_RANK];     // source strides
    int64_t dst_stride[HT_MAX_RANK];     // dest strides (in squashed source dim order)
    bool is_identity;                    // true when no actual data movement needed
};

// Squash adjacent dims that are contiguous in both src and dst strides.
// Delegates to layout::squash_dimensions_dual. Caches the identity check:
// src and dst strides have the same ordering iff no transpose is needed.
inline SquashResult squash_dims(
    const int32_t* dims,
    const int64_t* src_strides, const int64_t* dst_strides_for_src,
    int rank)
{
    auto sq = layout::squash_dimensions_dual<HT_MAX_RANK>(
        dims, src_strides, dst_strides_for_src, rank);

    SquashResult r{};
    r.rank = sq.rank;
    if (!sq.fits()) {
        // More than HT_MAX_RANK non-mergeable dims: unrepresentable. Arrays
        // stay unwritten; build_task_for_shape flags rank_overflow (NOT skip,
        // which would silently leave the data untransposed).
        return r;
    }
    for (int i = 0; i < sq.rank; ++i) {
        r.shape[i] = static_cast<int32_t>(sq.shapes[i]);
        r.src_stride[i] = sq.strides_a[i];
        r.dst_stride[i] = sq.strides_b[i];
    }

    // Identity check: src and dst strides have the same relative ordering
    r.is_identity = true;
    for (int d = 0; d + 1 < r.rank; ++d) {
        bool src_order = (r.src_stride[d] >= r.src_stride[d + 1]);
        bool dst_order = (r.dst_stride[d] >= r.dst_stride[d + 1]);
        if (src_order != dst_order) { r.is_identity = false; break; }
    }

    return r;
}

// Helper: Find the largest integer 'x' (where x <= max_val) that perfectly divides 'shape'.
// If no perfect divisor exists in the upper ranges, returns a power-of-2 or max_val.
inline int find_best_tile_dim(int shape, int max_val) {
    if (shape <= max_val) return shape; // Fits perfectly

    // 1. Search for a perfect divisor (greedily starting from max_val down to 1)
    // We only search down to max_val/3 to avoid shrinking the tile too much.
    for (int i = max_val; i >= std::max(1, max_val / 3); --i) {
        if (shape % i == 0) return i;
    }

    // 2. Fallback: If no perfect divisor exists, prefer multiples of 8 or 16 for memory alignment
    int aligned = (max_val / 8) * 8;
    if (aligned > 0) return aligned;

    // 3. Last resort
    return max_val;
}

// Fast fixed-point log2: ilog2(x) × 256 + linear interpolation.
// Used by both tilers for scoring in integer arithmetic.
inline int flog2_fp(int x) {
    if (x <= 1) return 0;
    int bits = 31 - __builtin_clz(x);
    int lo = 1 << bits;
    int frac = ((x - lo) << 8) / lo;
    return (bits << 8) + frac;
}

// Coalescing-aware tiler: Phase 0 (shared-tail) + unified greedy growth.
//
// Scoring model:
//   score(t) = log₂(min_eff) + α₂·log₂(max_eff) − λ·log₂(num_tiles)
//
// where ε(τ) = min(τ, W)/W is coalescing efficiency (W=32 warp width),
// and num_tiles = ∏ ceil(s_i / t_i) is the total tile count.
//
// Tile count replaces both edge-waste and volume terms: fewer tiles means
// less edge-tile overhead (divergence, register-map loss) AND better
// amortization of per-tile SMEM/sync cost.  For small shapes where edge
// tiles dominate, the log(tiles) penalty naturally scales up.
//
// λ ≈ 2.5: per-tile overhead relative to coalescing benefit.
// α₂ = 0.25: secondary weight on the non-bottleneck direction.
inline void optimize_tile_shape(
    int32_t squashed_rank,
    const int32_t* full_shape,
    const int64_t* src_stride,
    const int64_t* dst_stride,
    int32_t tile_vol_limit,
    int32_t* tile_shape_out)
{
    int32_t read_order[HT_MAX_RANK], write_order[HT_MAX_RANK];
    std::iota(read_order, read_order + squashed_rank, 0);
    std::sort(read_order, read_order + squashed_rank,
              [&](int a, int b) { return src_stride[a] < src_stride[b]; });
    std::iota(write_order, write_order + squashed_rank, 0);
    std::sort(write_order, write_order + squashed_rank,
              [&](int a, int b) { return dst_stride[a] < dst_stride[b]; });

    for (int d = 0; d < squashed_rank; ++d)
        tile_shape_out[d] = 1;

    int budget = tile_vol_limit;

    // Phase 0: Shared-tail pre-allocation.
    // Dims at the same innermost positions in both raster orders benefit
    // both directions equally — full-extent tiling is provably optimal.
    for (int k = 0; k < squashed_rank && budget > 1; ++k) {
        if (read_order[k] != write_order[k]) break;
        int d = read_order[k];
        int cap = std::min(full_shape[d], budget);
        tile_shape_out[d] = find_best_tile_dim(full_shape[d], cap);
        budget /= tile_shape_out[d];
    }

    // Phase 1: Seed contested innermost dims.
    // The growth loop optimizes min(ε_r, ε_w) which requires both
    // directions to have nonzero coalescing.  Seed conservatively to
    // W/2 = 16 so the growth loop has budget to find the right balance
    // between coalescing and tile count.
    for (int pass = 0; pass < 2 && budget > 1; ++pass) {
        const int32_t* order = (pass == 0) ? read_order : write_order;
        for (int k = 0; k < squashed_rank && budget > 1; ++k) {
            int d = order[k];
            if (tile_shape_out[d] > 1) continue;
            int seed = std::min(full_shape[d], std::min(16, budget));
            tile_shape_out[d] = seed;
            budget /= seed;
            break;
        }
    }

    // Helpers
    auto tile_product = [&]() {
        int p = 1;
        for (int d = 0; d < squashed_rank; ++d) p *= tile_shape_out[d];
        return p;
    };

    auto coalesced_tail = [&](const int32_t* order) {
        int tail = 1;
        for (int k = 0; k < squashed_rank; ++k) {
            int d = order[k];
            tail *= tile_shape_out[d];
            if (tile_shape_out[d] < full_shape[d]) break;
        }
        return tail;
    };

    // Tile count for a single dim
    auto dim_tiles = [](int shape, int tile) -> int {
        return (shape + tile - 1) / tile;
    };

    // Scoring coefficients (fixed-point × 256)
    constexpr int LAMBDA = 640;  // 2.5 × 256: tile-count penalty
    constexpr int ALPHA2 = 64;   // 0.25 × 256: max-tail weight

    // Unified growth: iteratively pick the best (dim, value) pair.
    // With doubling, need O(R × log₂(T)) iterations. R=8, log₂(1024)=10 → 80.
    for (int iter = 0; iter < squashed_rank * 12; ++iter) {
        int cur_prod = tile_product();
        if (cur_prod >= tile_vol_limit) break;

        int cur_rt = coalesced_tail(read_order);
        int cur_wt = coalesced_tail(write_order);

        int eff_rt = flog2_fp(std::min(cur_rt, 32));
        int eff_wt = flog2_fp(std::min(cur_wt, 32));
        int cur_eff = std::min(eff_rt, eff_wt) * 256
                    + ALPHA2 * std::max(eff_rt, eff_wt);
        int cur_tiles_log = 0;
        for (int d = 0; d < squashed_rank; ++d)
            cur_tiles_log += flog2_fp(dim_tiles(full_shape[d], tile_shape_out[d]));

        int best_dim = -1, best_val = 0, best_score = 0;

        for (int d = 0; d < squashed_rank; ++d) {
            if (tile_shape_out[d] >= full_shape[d]) continue;

            int max_fit = std::min(full_shape[d],
                tile_vol_limit / (cur_prod / tile_shape_out[d]));

            // Candidates: doubling (at most 2× current tile), divisor
            // near that point, and full extent.  Doubling prevents the
            // greedy from overshooting one dim and starving others.
            int cands[4];
            int nc = 0;
            int dbl = std::min(tile_shape_out[d] * 2, max_fit);
            if (dbl > tile_shape_out[d]) {
                cands[nc++] = dbl;
                int dv = find_best_tile_dim(full_shape[d], dbl);
                if (dv > tile_shape_out[d] && dv != dbl)
                    cands[nc++] = dv;
            }
            if (full_shape[d] <= max_fit && full_shape[d] > tile_shape_out[d])
                cands[nc++] = full_shape[d];

            int old_tiles_d = flog2_fp(dim_tiles(full_shape[d], tile_shape_out[d]));

            for (int ci = 0; ci < nc; ++ci) {
                int v = cands[ci];
                if (v <= tile_shape_out[d] || v > max_fit) continue;

                int old_val = tile_shape_out[d];
                tile_shape_out[d] = v;
                int new_rt = coalesced_tail(read_order);
                int new_wt = coalesced_tail(write_order);
                tile_shape_out[d] = old_val;

                int n_eff_rt = flog2_fp(std::min(new_rt, 32));
                int n_eff_wt = flog2_fp(std::min(new_wt, 32));
                int new_eff = std::min(n_eff_rt, n_eff_wt) * 256
                            + ALPHA2 * std::max(n_eff_rt, n_eff_wt);
                int new_tiles_d = flog2_fp(dim_tiles(full_shape[d], v));
                int new_tiles_log = cur_tiles_log - old_tiles_d + new_tiles_d;

                // score = Δ(coalescing) − λ·Δlog(tiles)
                // Both terms in flog2_fp units (×256), no extra scaling.
                int score = (new_eff - cur_eff)
                          - LAMBDA * (new_tiles_log - cur_tiles_log);

                if (score > best_score) {
                    best_dim = d;
                    best_val = v;
                    best_score = score;
                }
            }
        }

        if (best_dim < 0) break;
        tile_shape_out[best_dim] = best_val;
    }
}

// Exhaustive-search tiler: evaluates all feasible tile combinations and
// picks the one maximizing a bandwidth-model score.  Same signature as
// optimize_tile_shape so callers can swap freely.
//
// score(t) = α·log₂(min_tail) + 0.25·log₂(max_tail)
//          + β·log₂(volume)   − γ·edge_waste
//
// Uses two-regime effective tail: linear ≤32 (warp coalescing),
// sqrt beyond 32 (DRAM row locality).
//
// Scoring uses fixed-point ×256 arithmetic for speed; specialized flat
// loops for R=2 and R=3, greedy fallback for R≥4.

// Fixed-point effective-tail log2: log2(eff_tail(raw)) × 256
inline int eff_tail_log_fp(int raw) {
    if (raw <= 32) return flog2_fp(raw);
    return 640 + (flog2_fp(raw) >> 1);
}

// Generate tile candidates for a single dim extent.
// Returns count; fills cands_out (ascending, deduped, at most max_cands).
inline int gen_tile_candidates(int shape, int budget, int* cands_out, int max_cands) {
    int n = 0;
    int cap = std::min(shape, budget);
    // Reserve one slot for full extent (always important for coalescing)
    int loop_limit = max_cands - 1;
    for (int p = 1; p <= cap; p *= 2) {
        int v = std::min(shape, p);
        cands_out[n++] = v;
        if (shape > v && shape % v != 0 && n < loop_limit) {
            for (int delta = -2; delta <= 2; ++delta) {
                int dv = v + delta;
                if (dv > 0 && dv <= cap && dv != v && shape % dv == 0) {
                    cands_out[n++] = dv;
                    break;
                }
            }
        }
        if (n >= loop_limit) break;
    }
    if (shape <= cap) cands_out[n++] = shape;  // always include full extent
    std::sort(cands_out, cands_out + n);
    int u = 0;
    for (int i = 0; i < n; ++i)
        if (i == 0 || cands_out[i] != cands_out[i - 1])
            cands_out[u++] = cands_out[i];
    return u;
}

inline void optimize_tile_shape_search(
    int32_t squashed_rank,
    const int32_t* full_shape,
    const int64_t* src_stride,
    const int64_t* dst_stride,
    int32_t tile_vol_limit,
    int32_t* tile_shape_out)
{
    // For R≥4 the search space is too large; fall back to greedy.
    if (squashed_rank >= 4) {
        optimize_tile_shape(squashed_rank, full_shape, src_stride, dst_stride,
                            tile_vol_limit, tile_shape_out);
        return;
    }

    constexpr int FP = 8;
    constexpr int ALPHA = 256;
    constexpr int ALPHA2 = 64;
    constexpr int BETA  = 38;
    constexpr int GAMMA = 768;

    // For R=2, use a tight flat loop with no allocations or sorts on orders.
    if (squashed_rank == 2) {
        // Determine read/write innermost dim (stride-based)
        int rd0 = (src_stride[0] < src_stride[1]) ? 0 : 1;
        int wd0 = (dst_stride[0] < dst_stride[1]) ? 0 : 1;

        constexpr int MC = 16;
        int c0[MC], c1[MC];
        int n0 = gen_tile_candidates(full_shape[0], tile_vol_limit, c0, MC);
        int n1 = gen_tile_candidates(full_shape[1], tile_vol_limit, c1, MC);

        int best_score = INT_MIN;
        tile_shape_out[0] = 1;
        tile_shape_out[1] = 1;

        for (int i = 0; i < n0; ++i) {
            int t0 = c0[i];
            int budget1 = tile_vol_limit / t0;
            int ew0_num = ((full_shape[0] + t0 - 1) / t0) * t0 - full_shape[0];
            int ew0_den = ((full_shape[0] + t0 - 1) / t0) * t0;
            int vl0 = (BETA * flog2_fp(t0)) >> FP;

            for (int j = 0; j < n1; ++j) {
                int t1 = c1[j];
                if (t1 > budget1) break;  // ascending: rest are larger

                // Coalesced tails for R=2: tail = innermost tile, or full product if innermost is full
                int rt, wt;
                {
                    int inner = (rd0 == 0) ? t0 : t1;
                    int outer = (rd0 == 0) ? t1 : t0;
                    rt = (inner < full_shape[rd0]) ? inner : inner * outer;
                }
                {
                    int inner = (wd0 == 0) ? t0 : t1;
                    int outer = (wd0 == 0) ? t1 : t0;
                    wt = (inner < full_shape[wd0]) ? inner : inner * outer;
                }

                int ert = eff_tail_log_fp(rt);
                int ewt = eff_tail_log_fp(wt);
                int min_tl = std::min(ert, ewt);
                int max_tl = std::max(ert, ewt);

                int ew1_num = ((full_shape[1] + t1 - 1) / t1) * t1 - full_shape[1];
                int ew1_den = ((full_shape[1] + t1 - 1) / t1) * t1;
                int edge = (GAMMA * ew0_num * 256) / ew0_den
                         + (GAMMA * ew1_num * 256) / ew1_den;
                edge >>= FP;

                int vl = vl0 + ((BETA * flog2_fp(t1)) >> FP);
                int sc = ((ALPHA * min_tl + ALPHA2 * max_tl) >> FP) + vl - edge;

                if (sc > best_score) {
                    best_score = sc;
                    tile_shape_out[0] = t0;
                    tile_shape_out[1] = t1;
                }
            }
        }
        return;
    }

    // For R=3, flat triple loop with the same scoring.
    if (squashed_rank == 3) {
        // Sort dims by src/dst stride to find read/write orders
        int ro[3], wo[3];
        for (int i = 0; i < 3; ++i) ro[i] = wo[i] = i;
        // Simple insertion sort for 3 elements
        if (src_stride[ro[1]] < src_stride[ro[0]]) std::swap(ro[0], ro[1]);
        if (src_stride[ro[2]] < src_stride[ro[1]]) { std::swap(ro[1], ro[2]);
            if (src_stride[ro[1]] < src_stride[ro[0]]) std::swap(ro[0], ro[1]); }
        if (dst_stride[wo[1]] < dst_stride[wo[0]]) std::swap(wo[0], wo[1]);
        if (dst_stride[wo[2]] < dst_stride[wo[1]]) { std::swap(wo[1], wo[2]);
            if (dst_stride[wo[1]] < dst_stride[wo[0]]) std::swap(wo[0], wo[1]); }

        constexpr int MC = 8;
        int c[3][MC];
        int nc[3];
        for (int d = 0; d < 3; ++d)
            nc[d] = gen_tile_candidates(full_shape[d], tile_vol_limit, c[d], MC);

        int best_score = INT_MIN;
        tile_shape_out[0] = tile_shape_out[1] = tile_shape_out[2] = 1;

        for (int i0 = 0; i0 < nc[0]; ++i0) {
            int t0 = c[0][i0];
            int budget1 = tile_vol_limit / t0;
            int ew0_num = ((full_shape[0] + t0 - 1) / t0) * t0 - full_shape[0];
            int ew0_den = ((full_shape[0] + t0 - 1) / t0) * t0;
            int vl0 = (BETA * flog2_fp(t0)) >> FP;

            for (int i1 = 0; i1 < nc[1]; ++i1) {
                int t1 = c[1][i1];
                if (t1 > budget1) break;
                int budget2 = budget1 / t1;
                int ew1_num = ((full_shape[1] + t1 - 1) / t1) * t1 - full_shape[1];
                int ew1_den = ((full_shape[1] + t1 - 1) / t1) * t1;
                int vl01 = vl0 + ((BETA * flog2_fp(t1)) >> FP);

                for (int i2 = 0; i2 < nc[2]; ++i2) {
                    int t2 = c[2][i2];
                    if (t2 > budget2) break;

                    int tile[3] = {t0, t1, t2};

                    // Coalesced tails
                    auto coal_tail = [&](const int* order) {
                        int tail = 1;
                        for (int k = 0; k < 3; ++k) {
                            int d = order[k];
                            tail *= tile[d];
                            if (tile[d] < full_shape[d]) break;
                        }
                        return tail;
                    };
                    int rt = coal_tail(ro);
                    int wt = coal_tail(wo);

                    int ert = eff_tail_log_fp(rt);
                    int ewt = eff_tail_log_fp(wt);
                    int min_tl = std::min(ert, ewt);
                    int max_tl = std::max(ert, ewt);

                    int ew2_num = ((full_shape[2] + t2 - 1) / t2) * t2 - full_shape[2];
                    int ew2_den = ((full_shape[2] + t2 - 1) / t2) * t2;
                    int edge = ((GAMMA * ew0_num * 256) / ew0_den
                              + (GAMMA * ew1_num * 256) / ew1_den
                              + (GAMMA * ew2_num * 256) / ew2_den) >> FP;

                    int vl = vl01 + ((BETA * flog2_fp(t2)) >> FP);
                    int sc = ((ALPHA * min_tl + ALPHA2 * max_tl) >> FP) + vl - edge;

                    if (sc > best_score) {
                        best_score = sc;
                        tile_shape_out[0] = t0;
                        tile_shape_out[1] = t1;
                        tile_shape_out[2] = t2;
                    }
                }
            }
        }
        return;
    }
}

// Build rasterization order: sort dim indices by stride ascending (innermost first).
inline void build_raster_order(
    int squashed_rank, const int64_t* strides, int32_t* order_out)
{
    std::iota(order_out, order_out + squashed_rank, 0);
    std::sort(order_out, order_out + squashed_rank, [&](int a, int b) {
        return strides[a] < strides[b];
    });
}

// Raster-order squashing: merge adjacent dims in a given raster order when
// the inner dim is fully tiled and strides are contiguous.  Returns the
// squashed rank and populates the output arrays (divmod, stride, etc.).
struct RasterSquashResult {
    int rank;                              // squashed rank
    TileDivmod divmod[HT_MAX_RANK];        // tile divmods
    int64_t    stride[HT_MAX_RANK];        // global strides
    int32_t    smem_stride[HT_MAX_RANK];   // SMEM routing strides (write path only)
    uint16_t   inner_vol[HT_MAX_RANK];     // product of inner fulfilled tile shapes
    uint8_t    dim_idx[HT_MAX_RANK];       // outermost dim index
};

// Read-path raster squash: merge if contiguous in src AND inner dim fulfilled.
inline RasterSquashResult raster_squash_read(
    int squashed_rank,
    const int32_t* read_order,
    const int32_t* tile_shape,
    const int32_t* full_shape,
    const int64_t* src_stride)
{
    RasterSquashResult out{};
    if (squashed_rank == 0) { out.rank = 0; return out; }

    int d0 = read_order[0];
    int cur_tile = tile_shape[d0];
    int64_t cur_stride = src_stride[d0];
    int outermost = d0;
    uint16_t cur_inner_vol = 1;
    int rr = 0;

    for (int i = 1; i < squashed_rank; ++i) {
        int d = read_order[i];
        int prev_d = read_order[i - 1];

        bool contiguous_src = (src_stride[d] == src_stride[prev_d] * full_shape[prev_d]);
        bool fulfilled = (tile_shape[prev_d] == full_shape[prev_d]);

        if (contiguous_src && fulfilled) {
            cur_inner_vol *= static_cast<uint16_t>(tile_shape[prev_d]);
            cur_tile *= tile_shape[d];
            outermost = d;
        } else {
            out.divmod[rr] = make_tile_divmod(cur_tile);
            out.stride[rr] = cur_stride;
            out.inner_vol[rr] = cur_inner_vol;
            out.dim_idx[rr] = static_cast<uint8_t>(outermost);
            rr++;
            cur_tile = tile_shape[d];
            cur_stride = src_stride[d];
            outermost = d;
            cur_inner_vol = 1;
        }
    }
    out.divmod[rr] = make_tile_divmod(cur_tile);
    out.stride[rr] = cur_stride;
    out.inner_vol[rr] = cur_inner_vol;
    out.dim_idx[rr] = static_cast<uint8_t>(outermost);
    out.rank = rr + 1;
    return out;
}

// Write-path raster squash: merge if contiguous in dst AND smem AND inner fulfilled.
inline RasterSquashResult raster_squash_write(
    int squashed_rank,
    const int32_t* write_order,
    const int32_t* tile_shape,
    const int32_t* full_shape,
    const int64_t* dst_stride,
    const int32_t* smem_stride_dim)
{
    RasterSquashResult out{};
    if (squashed_rank == 0) { out.rank = 0; return out; }

    int d0 = write_order[0];
    int cur_tile = tile_shape[d0];
    int64_t cur_dst_stride = dst_stride[d0];
    int32_t cur_smem_stride = smem_stride_dim[d0];
    int outermost = d0;
    uint16_t cur_inner_vol = 1;
    int wr = 0;

    for (int i = 1; i < squashed_rank; ++i) {
        int d = write_order[i];
        int prev_d = write_order[i - 1];

        bool contiguous_dst = (dst_stride[d] == dst_stride[prev_d] * full_shape[prev_d]);
        bool contiguous_smem = (smem_stride_dim[d] == smem_stride_dim[prev_d] * tile_shape[prev_d]);
        bool fulfilled = (tile_shape[prev_d] == full_shape[prev_d]);

        if (contiguous_dst && fulfilled && contiguous_smem) {
            cur_inner_vol *= static_cast<uint16_t>(tile_shape[prev_d]);
            cur_tile *= tile_shape[d];
            outermost = d;
        } else {
            out.divmod[wr] = make_tile_divmod(cur_tile);
            out.stride[wr] = cur_dst_stride;
            out.smem_stride[wr] = cur_smem_stride;
            out.inner_vol[wr] = cur_inner_vol;
            out.dim_idx[wr] = static_cast<uint8_t>(outermost);
            wr++;
            cur_tile = tile_shape[d];
            cur_dst_stride = dst_stride[d];
            cur_smem_stride = smem_stride_dim[d];
            outermost = d;
            cur_inner_vol = 1;
        }
    }
    out.divmod[wr] = make_tile_divmod(cur_tile);
    out.stride[wr] = cur_dst_stride;
    out.smem_stride[wr] = cur_smem_stride;
    out.inner_vol[wr] = cur_inner_vol;
    out.dim_idx[wr] = static_cast<uint8_t>(outermost);
    out.rank = wr + 1;
    return out;
}

// Build a HypertileShapeMax for a single block shape.
// Returns {shape, info, skip}. skip=true if shape squashes to identity or rank<2.
// dims: block dimensions [rank], full_perm/inv_perm: full-rank permutation and its inverse.
struct SingleTaskResult {
    HypertileShapeMax shape;
    HypertileTaskInfo info;
    bool skip;
    // Squashed rank exceeds HT_MAX_RANK: the shape NEEDS a transpose but the
    // hypertile kernel cannot represent it. Callers must fail loudly — this
    // is distinct from skip (= no data movement needed).
    bool rank_overflow = false;
};

inline SingleTaskResult build_task_for_shape(
    const dim_t* dims, dim_t rank, dim_t /*head_rank*/,
    const int32_t* full_perm, const int32_t* inv_perm,
    int tile_vol_limit)
{
    SingleTaskResult result{};
    result.skip = true;

    int64_t block_vol = 1;
    for (int d = 0; d < rank; ++d) block_vol *= dims[d];
    if (block_vol == 0) return result;

    int64_t src_stride[16];
    src_stride[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
        src_stride[d] = src_stride[d + 1] * dims[d + 1];

    int64_t dst_strides_pos[16];
    dst_strides_pos[rank - 1] = 1;
    for (int dd = rank - 2; dd >= 0; --dd)
        dst_strides_pos[dd] = dst_strides_pos[dd + 1] * dims[inv_perm[dd + 1]];
    int64_t dst_stride_for_src[16];
    for (int sd = 0; sd < rank; ++sd)
        dst_stride_for_src[sd] = dst_strides_pos[full_perm[sd]];

    SquashResult sq = squash_dims(dims, src_stride, dst_stride_for_src, rank);
    if (sq.rank > HT_MAX_RANK) {
        result.rank_overflow = true;
        return result;
    }
    if (sq.rank < 2 || sq.is_identity) return result;

    int32_t tile_shape[HT_MAX_RANK];
    if (g_use_search_tiler)
        optimize_tile_shape_search(sq.rank, sq.shape, sq.src_stride, sq.dst_stride,
                                   tile_vol_limit, tile_shape);
    else
        optimize_tile_shape(sq.rank, sq.shape, sq.src_stride, sq.dst_stride,
                            tile_vol_limit, tile_shape);

    int32_t tile_vol = 1;
    for (int d = 0; d < sq.rank; ++d) tile_vol *= tile_shape[d];

    int32_t grid_shape[HT_MAX_RANK];
    uint32_t num_tiles = 1;
    for (int d = 0; d < sq.rank; ++d) {
        grid_shape[d] = (sq.shape[d] + tile_shape[d] - 1) / tile_shape[d];
        num_tiles *= static_cast<uint32_t>(grid_shape[d]);
    }

    int32_t read_order[HT_MAX_RANK], write_order[HT_MAX_RANK];
    build_raster_order(sq.rank, sq.src_stride, read_order);
    build_raster_order(sq.rank, sq.dst_stride, write_order);

    int32_t smem_rs_dim[HT_MAX_RANK] = {};
    int32_t rs = 1;
    for (int i = 0; i < sq.rank; ++i) {
        int d = read_order[i];
        smem_rs_dim[d] = rs;
        rs *= tile_shape[d];
    }

    HypertileShapeMax& shape = result.shape;
    shape = HypertileShapeMax{};
    shape.num_tiles = num_tiles;
    shape.tile_volume = static_cast<uint16_t>(tile_vol);
    shape.squashed_rank = static_cast<uint8_t>(sq.rank);

    uint8_t dim_to_grid[HT_MAX_RANK];
    std::memset(dim_to_grid, 0xFF, sizeof(dim_to_grid));
    int gr = 0;
    for (int d = sq.rank - 1; d >= 0; --d) {
        if (grid_shape[d] > 1) {
            shape.grid_divmod[gr] = make_grid_divmod(grid_shape[d]);
            shape.grid_src_step[gr] = static_cast<uint32_t>(tile_shape[d] * sq.src_stride[d]);
            shape.grid_dst_step[gr] = static_cast<uint32_t>(tile_shape[d] * sq.dst_stride[d]);
            shape.edge_sizes[gr] = static_cast<uint16_t>(sq.shape[d] % tile_shape[d]);
            dim_to_grid[d] = static_cast<uint8_t>(gr);
            gr++;
        }
    }
    shape.GR = static_cast<uint8_t>(gr);

    RasterSquashResult rsq_r = raster_squash_read(
        sq.rank, read_order, tile_shape, sq.shape, sq.src_stride);
    shape.read.RR = static_cast<uint8_t>(rsq_r.rank);
    for (int i = 0; i < rsq_r.rank; ++i) {
        shape.read.divmod[i] = rsq_r.divmod[i];
        shape.read.src_stride[i] = static_cast<uint32_t>(rsq_r.stride[i]);
        uint8_t gi = dim_to_grid[rsq_r.dim_idx[i]];
        shape.read.grid_idx[i] = gi;
        if (gi < gr && shape.edge_sizes[gi] != 0)
            shape.read.edge_allowed[i] = static_cast<uint16_t>(
                shape.edge_sizes[gi] * rsq_r.inner_vol[i]);
        else
            shape.read.edge_allowed[i] = 0xFFFF;
    }

    RasterSquashResult rsq_w = raster_squash_write(
        sq.rank, write_order, tile_shape, sq.shape,
        sq.dst_stride, smem_rs_dim);
    shape.WR = static_cast<uint8_t>(rsq_w.rank);
    for (int i = 0; i < rsq_w.rank; ++i) {
        shape.write.divmod[i] = rsq_w.divmod[i];
        shape.write.dst_stride[i] = static_cast<uint32_t>(rsq_w.stride[i]);
        shape.write.smem_stride[i] = static_cast<uint16_t>(rsq_w.smem_stride[i]);
        uint8_t gi = dim_to_grid[rsq_w.dim_idx[i]];
        shape.write.grid_idx[i] = gi;
        if (gi < gr && shape.edge_sizes[gi] != 0)
            shape.write.edge_allowed[i] = static_cast<uint16_t>(
                shape.edge_sizes[gi] * rsq_w.inner_vol[i]);
        else
            shape.write.edge_allowed[i] = 0xFFFF;
    }

    assert(shape.GR <= shape.read.RR && "GR must be <= RR");
    assert(shape.read.RR <= shape.WR && "RR must be <= WR");

    // Fill diagnostics
    result.info.squashed_rank = sq.rank;
    for (int d = 0; d < sq.rank; ++d) {
        result.info.tile_shape[d] = tile_shape[d];
        result.info.full_shape[d] = sq.shape[d];
        result.info.grid_shape[d] = grid_shape[d];
        result.info.read_order[d] = read_order[d];
        result.info.write_order[d] = write_order[d];
        result.info.src_stride[d] = sq.src_stride[d];
        result.info.dst_stride[d] = sq.dst_stride[d];
    }

    result.skip = false;
    return result;
}

// ============================================================================
// Device-side kernel
// ============================================================================

// Shared-memory bank-conflict avoidance strategy (compile-time toggle).
// Define HT_SMEM_MODE before including this header to select:
//   0 = linear padding:  addr = x + x/32  (needs TV + TV/32 buffer)
//   1 = XOR swizzle:     addr = x ^ ((x >> 5) & 31)  (needs TV buffer)
#ifndef HT_SMEM_MODE
#define HT_SMEM_MODE 1
#endif

#if HT_SMEM_MODE == 0
#define HT_SMEM_IDX(x)  ((x) + ((x) >> 5))
#define HT_SMEM_ELEMS(tv) ((tv) + ((tv) >> 5))
#else
#define HT_SMEM_IDX(x)  ((x) ^ (((x) >> 5) & 31))
#define HT_SMEM_ELEMS(tv) (tv)
#endif

// Binary search: find largest i such that arr[i] <= val.
__device__ inline int ht_upper_bound(const int32_t* arr, int n, int val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;  // largest index with arr[index] <= val
}

// ---------------------------------------------------------------------------
// HypertileLoader<R>: stateless read-path primitive.
// Takes a const ref to HypertileReadInfo<R> (a sub-struct of HypertileShape,
// or embedded in any other task type). Kernel owns register arrays.
// ---------------------------------------------------------------------------
template <int R>
struct HypertileLoader {
    const HypertileReadInfo<R>& info;

    // Compute read offset for linear tile index t.
    // Called once per map_id change to populate read_off[EPT].
    __device__ __forceinline__
    uint32_t compute_offset(int t) const {
        const int SRR = info.RR;
        uint32_t urem = static_cast<uint32_t>(t);
        uint32_t ro = 0;
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            if (i < SRR) {
                uint32_t q, r;
                if (i < R - 1) {
                    info.divmod[i].divmod(urem, q, r);
                    urem = q;
                } else {
                    r = urem;
                }
                ro += r * info.src_stride[i];
            }
        }
        return ro;
    }

    // Fast interior tile: load from scattered global into linear SMEM.
    // tv is tile_volume snapshotted into a register by the caller — avoids
    // SMEM reload after every STS (the compiler can't prove smem[] writes
    // don't alias the s_shape.tile_volume field in shared memory).
    template <typename T, int EPT, int BS>
    __device__ __forceinline__
    void load_fast(T* smem, const T* src, int64_t base_in,
                   const uint32_t read_off[EPT], int tv) const {
        #pragma unroll
        for (int step = 0; step < EPT; ++step) {
            int t = threadIdx.x + step * BS;
            if (t < tv) {
                smem[HT_SMEM_IDX(t)] = src[base_in + read_off[step]];
            }
        }
    }

    // Edge tile: load with bounds checking.
    // Dimension pruning: divmod chain stops at the highest edge dim
    // (warp-uniform skip). Per-thread early-out on valid=false.
    template <typename T, int EPT, int BS>
    __device__ __forceinline__
    void load_edge(T* smem, const T* src, int64_t base_in,
                   const uint32_t read_off[EPT],
                   uint8_t edge_bits, int tv) const {
        const int SRR = info.RR;

        // Highest read-squashed dim that needs bounds checking.
        // Dims beyond this need no divmod at all (warp-uniform skip).
        int max_edge_i = -1;
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            if (i < SRR && (edge_bits & (1u << info.grid_idx[i])))
                max_edge_i = i;
        }

        for (int step = 0; step < EPT; ++step) {
            int t = threadIdx.x + step * BS;
            if (t < tv) {
                bool valid = true;
                uint32_t urem = static_cast<uint32_t>(t);
                uint32_t q, r;
                #pragma unroll
                for (int i = 0; i < R; ++i) {
                    if (i <= max_edge_i && valid) {
                        if (i < R - 1) {
                            info.divmod[i].divmod(urem, q, r);
                            urem = q;
                        } else {
                            r = urem;
                        }
                        if (edge_bits & (1u << info.grid_idx[i]))
                            valid = (r < info.edge_allowed[i]);
                    }
                }
                T val = T(0);
                if (valid) val = src[base_in + read_off[step]];
                smem[HT_SMEM_IDX(t)] = val;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// HypertileStorer<R>: stateless write-path primitive.
// Takes a const ref to HypertileWriteInfo<R>.
// ---------------------------------------------------------------------------
template <int R>
struct HypertileStorer {
    const HypertileWriteInfo<R>& info;

    struct WriteResult {
        uint32_t offset;
        uint16_t smem_route;
    };

    // Compute write offset and SMEM route for linear tile index t.
    __device__ __forceinline__
    WriteResult compute_offset(int t) const {
        uint32_t urem = static_cast<uint32_t>(t);
        uint32_t wo = 0;
        int32_t si = 0;
        uint32_t q, r;
        #pragma unroll
        for (int i = 0; i < R - 1; ++i) {
            info.divmod[i].divmod(urem, q, r);
            urem = q;
            wo += r * info.dst_stride[i];
            si += static_cast<int32_t>(r) * info.smem_stride[i];
        }
        wo += urem * info.dst_stride[R - 1];
        si += static_cast<int32_t>(urem) * info.smem_stride[R - 1];
        return {wo, static_cast<uint16_t>(si)};
    }

    // Fast interior tile: store from SMEM (read-raster) to scattered global.
    template <typename T, int EPT, int BS>
    __device__ __forceinline__
    void store_fast(T* dst, const T* smem, int64_t base_out,
                    const uint32_t write_off[EPT],
                    const uint16_t smem_route[EPT], int tv) const {
        #pragma unroll
        for (int step = 0; step < EPT; ++step) {
            int t = threadIdx.x + step * BS;
            if (t < tv) {
                dst[base_out + write_off[step]] =
                    smem[HT_SMEM_IDX(smem_route[step])];
            }
        }
    }

    // Edge tile: store with bounds checking. Dimension pruning as in load_edge.
    template <typename T, int EPT, int BS>
    __device__ __forceinline__
    void store_edge(T* dst, const T* smem, int64_t base_out,
                    const uint32_t write_off[EPT],
                    const uint16_t smem_route[EPT],
                    uint8_t edge_bits, int tv) const {

        int max_edge_i = -1;
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            if (edge_bits & (1u << info.grid_idx[i]))
                max_edge_i = i;
        }

        for (int step = 0; step < EPT; ++step) {
            int t = threadIdx.x + step * BS;
            if (t < tv) {
                bool valid = true;
                uint32_t urem = static_cast<uint32_t>(t);
                uint32_t q, r;
                #pragma unroll
                for (int i = 0; i < R - 1; ++i) {
                    if (i <= max_edge_i && valid) {
                        info.divmod[i].divmod(urem, q, r);
                        urem = q;
                        if (edge_bits & (1u << info.grid_idx[i]))
                            valid = (r < info.edge_allowed[i]);
                    }
                }
                if (max_edge_i == R - 1 && valid &&
                    (edge_bits & (1u << info.grid_idx[R - 1])))
                    valid = (urem < info.edge_allowed[R - 1]);
                if (valid) {
                    dst[base_out + write_off[step]] =
                        smem[HT_SMEM_IDX(smem_route[step])];
                }
            }
        }
    }
};

template <typename T, int TV, int BS, int R>
__global__ __launch_bounds__(BS, 4)
void hypertile_transpose_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const HypertileShape<R>* __restrict__ shapes,
    const HypertileMem* __restrict__ mems,
    const int32_t* __restrict__ prefix_sum,
    int32_t num_tasks,
    int32_t total_tiles)
{
    constexpr int EPT = TV / BS;
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    __shared__ HypertileShape<R> s_shape;

    // Per-task mem fields broadcast via smem.
    __shared__ int64_t s_src_base, s_dst_base;

    // Block-wide tile variables: computed by thread 0, broadcast via smem.
    __shared__ int64_t s_base_in, s_base_out;
    __shared__ uint32_t s_grid_r[R];  // raw grid coordinates (GR entries)
    __shared__ bool s_interior;
    __shared__ uint8_t s_edge_bits;    // bit i set iff dim i is at edge position

    // Register maps (persist across tiles of same shape).
    uint32_t read_off[EPT];
    uint32_t write_off[EPT];
    uint16_t smem_route[EPT];

    // Stateless loader/storer primitives — bound to s_shape's sub-structs.
    HypertileLoader<R> loader{s_shape.read};
    HypertileStorer<R> storer{s_shape.write};

    int current_task = -1;
    int current_task_end_tile = -1;
    int c_prefix = 0;
    bool have_map = false;
    uint32_t prev_shape_idx = 0xFFFFFFFF;
    int tv = 0;  // tile_volume in register — avoids SMEM aliasing reload

    constexpr int CHUNK_SIZE = 16;

    for (int base_tile = blockIdx.x * CHUNK_SIZE; base_tile < total_tiles;
         base_tile += gridDim.x * CHUNK_SIZE)
    {
        for (int k = 0; k < CHUNK_SIZE; ++k) {
            int tg = base_tile + k;
            if (tg >= total_tiles) break;

            bool need_full_decomp = (k == 0);  // chunk start — no prior state
            if (tg >= current_task_end_tile) {
                need_full_decomp = true;  // task boundary resets grid
                current_task = ht_upper_bound(prefix_sum, num_tasks + 1, tg);
                current_task_end_tile = prefix_sum[current_task + 1];
                c_prefix = prefix_sum[current_task];

                // All threads read shape_idx from global (L1 cached, warp-uniform).
                // Thread 0 also loads src/dst base to shared memory.
                uint32_t cur_shape_idx = mems[current_task].shape_idx;
                if (threadIdx.x == 0) {
                    s_src_base = mems[current_task].src_base;
                    s_dst_base = mems[current_task].dst_base;
                }

                // Load shape only if shape_idx changed
                if (cur_shape_idx != prev_shape_idx) {
                    prev_shape_idx = cur_shape_idx;
                    static_assert(sizeof(HypertileShape<R>) % sizeof(uint32_t) == 0,
                                  "HypertileShape must be 4-byte aligned");
                    constexpr int SHAPE_WORDS = sizeof(HypertileShape<R>) / sizeof(uint32_t);
                    auto* dst_w = reinterpret_cast<uint32_t*>(&s_shape);
                    const auto* src_p = reinterpret_cast<const uint32_t*>(&shapes[cur_shape_idx]);
                    for (int w = threadIdx.x; w < SHAPE_WORDS; w += BS)
                        dst_w[w] = src_p[w];
                    have_map = false;  // force register map recompute
                }
                __syncthreads();

                // Snapshot tile_volume into a register once after shape load.
                // s_shape.tile_volume lives in shared memory; without this,
                // the compiler must reload it after every STS to smem[]
                // because it can't prove the store doesn't alias tile_volume.
                tv = s_shape.tile_volume;

                if (!have_map) {
                    have_map = true;

                    #pragma unroll
                    for (int step = 0; step < EPT; ++step) {
                        int t = threadIdx.x + step * BS;
                        read_off[step] = loader.compute_offset(t);
                        auto wr = storer.compute_offset(t);
                        write_off[step] = wr.offset;
                        smem_route[step] = wr.smem_route;
                    }
                }
            }

            const int SGR = s_shape.GR;

            if (threadIdx.x == 0) {
                int64_t bi, bo;
                bool intr;

                if (need_full_decomp) {
                    // Full 1D → ND decomposition (chunk start or task boundary).
                    // Edge check: dim is at edge iff edge_sizes != 0 and coord
                    // is at the last position (r == divisor - 1).
                    uint32_t urem = static_cast<uint32_t>(tg - c_prefix);
                    bi = s_src_base;
                    bo = s_dst_base;
                    uint8_t eb = 0;

                    #pragma unroll
                    for (int i = 0; i < R; ++i) {
                        if (i < SGR) {
                            uint32_t q, r;
                            s_shape.grid_divmod[i].divmod(urem, q, r);
                            s_grid_r[i] = r;
                            bi += static_cast<int64_t>(r) * s_shape.grid_src_step[i];
                            bo += static_cast<int64_t>(r) * s_shape.grid_dst_step[i];
                            if (s_shape.edge_sizes[i] &&
                                r + 1 == s_shape.grid_divmod[i].divisor)
                                eb |= (1u << i);
                            urem = q;
                        }
                    }
                    s_edge_bits = eb;
                    intr = (eb == 0);
                } else {
                    // Odometer increment: advance innermost grid coordinate by 1.
                    // Avoids GR divmod ops; common case (no carry) is 1 add + 1 cmp.
                    bi = s_base_in + s_shape.grid_src_step[0];
                    bo = s_base_out + s_shape.grid_dst_step[0];
                    s_grid_r[0]++;

                    if (s_grid_r[0] < s_shape.grid_divmod[0].divisor) {
                        // No carry (common case): only dim 0 changed.
                        uint8_t eb = s_edge_bits & static_cast<uint8_t>(~1u);
                        if (s_shape.edge_sizes[0] &&
                            s_grid_r[0] + 1 == s_shape.grid_divmod[0].divisor)
                            eb |= 1u;
                        s_edge_bits = eb;
                        intr = (eb == 0);
                    } else {
                        // Carry chain (rare: only when innermost dim wraps)
                        #pragma unroll
                        for (int i = 0; i < R; ++i) {
                            if (i >= SGR) break;
                            if (s_grid_r[i] < s_shape.grid_divmod[i].divisor) break;
                            bi -= static_cast<int64_t>(s_grid_r[i]) * s_shape.grid_src_step[i];
                            bo -= static_cast<int64_t>(s_grid_r[i]) * s_shape.grid_dst_step[i];
                            s_grid_r[i] = 0;
                            if (i + 1 < SGR) {  // GR ≤ R, so i+1 < SGR implies i+1 < R
                                s_grid_r[i+1]++;
                                bi += s_shape.grid_src_step[i+1];
                                bo += s_shape.grid_dst_step[i+1];
                            }
                        }
                        // Rescan all dims after carry
                        uint8_t eb = 0;
                        #pragma unroll
                        for (int i = 0; i < R; ++i) {
                            if (i < SGR && s_shape.edge_sizes[i] &&
                                s_grid_r[i] + 1 == s_shape.grid_divmod[i].divisor)
                                eb |= (1u << i);
                        }
                        s_edge_bits = eb;
                        intr = (eb == 0);
                    }
                }
                s_base_in = bi;
                s_base_out = bo;
                s_interior = intr;
            }
            __syncthreads();

            int64_t base_in = s_base_in;
            int64_t base_out = s_base_out;
            bool interior = s_interior;

            if (interior) {
                loader.template load_fast<T, EPT, BS>(smem, src, base_in, read_off, tv);
                __syncthreads();
                storer.template store_fast<T, EPT, BS>(dst, smem, base_out, write_off, smem_route, tv);
                __syncthreads();
            } else {
                loader.template load_edge<T, EPT, BS>(smem, src, base_in, read_off, s_edge_bits, tv);
                __syncthreads();
                storer.template store_edge<T, EPT, BS>(dst, smem, base_out, write_off, smem_route, s_edge_bits, tv);
                __syncthreads();
            }
        }
    }
}

// Dispatch helper: launch hypertile kernel with compile-time rank R.
// Launch hypertile kernel for a single rank slice.
// d_shapes/d_mems are type-erased (void*) — cast inside.
// Grid size is computed via occupancy query; smem is derived from TV.
template <typename T, int TV, int BS>
void launch_hypertile_kernel(
    const T* d_src, T* d_dst,
    const void* d_shapes,
    const void* d_mems,
    const int32_t* d_prefix_sum,
    int32_t num_tasks, int32_t total_tiles,
    int32_t rank,
    cudaStream_t stream)
{
    size_t smem_bytes = HT_SMEM_ELEMS(TV) * sizeof(T);

#define HT_LAUNCH_R(R_VAL) { \
    auto kern = hypertile_transpose_kernel<T, TV, BS, R_VAL>; \
    int grid = persistent_grid_size(kern, BS, static_cast<int>(smem_bytes), \
                                    static_cast<int>(total_tiles)); \
    kern<<<grid, BS, smem_bytes, stream>>>( \
            d_src, d_dst, \
            reinterpret_cast<const HypertileShape<R_VAL>*>(d_shapes), \
            reinterpret_cast<const HypertileMem*>(d_mems), \
            d_prefix_sum, num_tasks, total_tiles); \
    }

    switch (rank) {
        case 2: HT_LAUNCH_R(2); break;
        case 3: HT_LAUNCH_R(3); break;
        case 4: HT_LAUNCH_R(4); break;
        case 5: HT_LAUNCH_R(5); break;
        case 6: HT_LAUNCH_R(6); break;
        case 7: HT_LAUNCH_R(7); break;
        case 8: HT_LAUNCH_R(8); break;
        default: HT_LAUNCH_R(8); break;
    }
#undef HT_LAUNCH_R
}

// ============================================================================
// HypertileIndex: pre-computed shape metadata for cached transpose precomp.
// Built once per operand during pipeline_allocate. Per-batch, only lightweight
// HypertileMem + prefix_sum are assembled directly from the index.
// ============================================================================

struct HypertileIndexEntry {
    uint8_t  rank;        // WR (write-squashed rank = kernel template R)
    uint8_t  slice_idx;   // index into slices[] (which rank group)
    uint32_t shape_idx;   // index into that slice's d_shapes array
    uint32_t num_tiles;   // tile count for this shape
    bool     trivial;     // skip (identity after squash)
};

struct HypertileIndex {
    // Per shape_map template: indexed by shape_id (0..shape_map.num_unique-1)
    std::vector<HypertileIndexEntry> per_shape;

    // Per-rank-group metadata (one per unique WR value)
    struct SliceInfo {
        int rank = 0;                        // WR (kernel template R)
        int num_shapes = 0;                  // deduplicated shape count
        size_t shape_bytes = 0;              // total bytes for all shapes
        std::vector<uint8_t> h_shape_bytes;  // host-side packed HypertileShape<R>[]
        std::vector<uint32_t> h_num_tiles;    // num_tiles per shape (indexed by shape_idx)
    };
    int num_slices = 0;
    SliceInfo slices[HT_MAX_RANK];

    // Diagnostics (from shape templates)
    int32_t max_squashed_rank = 0;
    uint16_t max_tile_volume = 0;
    int32_t min_rr = HT_MAX_RANK, min_wr = HT_MAX_RANK;

    bool valid = false;
    // Some shape needs a transpose the kernel cannot represent (squashed rank
    // > HT_MAX_RANK). pipeline_prepare fails loudly on this — proceeding
    // would silently skip the transpose and corrupt results.
    bool rank_overflow = false;

    // No GPU resources owned (shapes live in TransposePrecompBuffers)
    ~HypertileIndex() = default;
    HypertileIndex() = default;
    HypertileIndex(HypertileIndex&&) = default;
    HypertileIndex& operator=(HypertileIndex&&) = default;
    HypertileIndex(const HypertileIndex&) = delete;
    HypertileIndex& operator=(const HypertileIndex&) = delete;
};

} // namespace bsptc

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
