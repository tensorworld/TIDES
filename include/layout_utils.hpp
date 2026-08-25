#pragma once

#include "types.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
#include <stdexcept>

// Layout/stride helpers shared by dense and block views.
namespace bsptc {
namespace layout {

using LayoutKind = LayoutType;

// ------------------------------------------------------------
// Canonical stride computation
// ------------------------------------------------------------

// Compute strides for a given layout (row/col major variants).
// Pseudocode: starting from contiguous side, accumulate product of later dims.
// Example shape [2,3,4]:
//   RowMajor -> strides [12,4,1]; ColumnMajor -> [1,2,6]
inline std::vector<std::int64_t>
compute_strides(const std::vector<std::int64_t>& shape,
                LayoutType layout)
{
    const size_t ndim = shape.size();
    if (ndim == 0)
        throw std::runtime_error("compute_strides: empty shape");

    std::vector<std::int64_t> strides(ndim, 0);

    switch (layout) {

        case LayoutType::RowMajor:
        case LayoutType::RightMajorStride: {
            // Rightmost index is contiguous
            strides[ndim - 1] = 1;
            for (size_t i = ndim - 1; i-- > 0; ) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            break;
        }

        case LayoutType::ColumnMajor:
        case LayoutType::LeftMajorStride: {
            // Leftmost index is contiguous
            strides[0] = 1;
            for (size_t i = 1; i < ndim; ++i) {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
            break;
        }

        default:
            throw std::runtime_error("compute_strides: unknown layout");
    }

    return strides;
}

// ------------------------------------------------------------
// Linear index computation
// ------------------------------------------------------------

inline std::int64_t
linear_index(const std::vector<std::int64_t>& indices,
             const std::vector<std::int64_t>& strides)
{
    if (indices.size() != strides.size())
        throw std::runtime_error("linear_index: rank mismatch");

    std::int64_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

// ------------------------------------------------------------
// Validation helpers
// ------------------------------------------------------------

inline bool
is_contiguous(const std::vector<std::int64_t>& shape,
              const std::vector<std::int64_t>& strides,
              LayoutType layout)
{
    auto canonical = compute_strides(shape, layout);
    return canonical == strides;
}

// ------------------------------------------------------------
// Block-level stride computation
// (used by block-sparse tensors)
// ------------------------------------------------------------

inline std::vector<std::int64_t>
compute_block_strides(const std::vector<std::int64_t>& block_shape,
                      LayoutType layout)
{
    return compute_strides(block_shape, layout);
}

// ------------------------------------------------------------
// Human-readable names
// ------------------------------------------------------------
inline const char* layout_name(LayoutType layout) {
    switch (layout) {
        case LayoutType::RowMajor: return "RowMajor";
        case LayoutType::ColumnMajor: return "ColumnMajor";
        case LayoutType::LeftMajorStride: return "LeftMajorStride";
        case LayoutType::RightMajorStride: return "RightMajorStride";
        default: return "Unknown";
    }
}

// ------------------------------------------------------------
// Dimension squashing (coalescing)
// ------------------------------------------------------------
//
// Merges adjacent dimensions that are contiguous in memory across all stride
// sets. Merge condition for each stride set s:
//   stride_s[outer] == shape[inner] * stride_s[inner]
// Skips size-1 dimensions. All outputs are stack-allocated (no heap).
//
// Single unified implementation parameterized on NStrides (1 or 2).
// Public API: squash_dimensions<MaxRank>()       — 1 stride set
//             squash_dimensions_dual<MaxRank>()  — 2 stride sets

// Stack-allocated squash result for 1 stride set.
// rank is the TRUE squashed rank and may exceed MaxRank; only the outermost
// min(rank, MaxRank) groups are written. Callers must reject results where
// !fits() — the arrays never hold more than MaxRank valid groups.
template <int MaxRank>
struct SquashedDims {
    int rank = 0;
    int64_t shapes[MaxRank];
    int64_t strides[MaxRank];
    bool fits() const { return rank <= MaxRank; }
};

// Stack-allocated squash result for 2 stride sets.
template <int MaxRank>
struct SquashedDimsDual {
    int rank = 0;
    int64_t shapes[MaxRank];
    int64_t strides_a[MaxRank];
    int64_t strides_b[MaxRank];
    bool fits() const { return rank <= MaxRank; }
};

namespace detail {

// Max input rank for the internal temporary buffer.
static constexpr int SQUASH_MAX_INPUT_RANK = 16;

// Core squash implementation for NStrides stride sets.
// Iterates inner-to-outer, skips size-1, merges contiguous groups.
// Writes at most max_out groups into out_shapes/out_strides in outer-to-inner
// order. Returns the TRUE squashed rank, which may exceed max_out — the
// writes are clamped so overflow can never corrupt the output arrays, even
// in NDEBUG builds. Callers detect overflow via rank > max_out (fits()).
template <int NStrides>
inline int squash_impl(
    const int64_t* shapes, const int64_t* const (&strides)[NStrides],
    int rank, int max_out,
    int64_t* out_shapes, int64_t* (&out_strides)[NStrides])
{
    assert(rank <= SQUASH_MAX_INPUT_RANK);

    if (rank <= 0) {
        out_shapes[0] = 1;
        for (int s = 0; s < NStrides; ++s)
            out_strides[s][0] = 0;
        return 1;
    }

    // Build coalesced groups in reverse (inner-to-outer), then reverse.
    int64_t tmp_shapes[SQUASH_MAX_INPUT_RANK];
    int64_t tmp_strides[NStrides][SQUASH_MAX_INPUT_RANK];
    int n = 0;

    int d = rank - 1;
    while (d >= 0) {
        int64_t cur_shape = shapes[d];
        int64_t cur_str[NStrides];
        for (int s = 0; s < NStrides; ++s)
            cur_str[s] = strides[s][d];

        // Skip size-1 dimensions
        if (cur_shape == 1 && rank > 1) { --d; continue; }

        // Try to merge with outer (lower-index) dimensions
        while (d > 0) {
            int64_t outer_shape = shapes[d - 1];
            if (outer_shape == 1) { --d; continue; }

            bool can_merge = true;
            for (int s = 0; s < NStrides; ++s)
                can_merge &= (strides[s][d - 1] == cur_shape * cur_str[s]);

            if (can_merge) {
                cur_shape *= outer_shape;
                --d;
            } else {
                break;
            }
        }

        tmp_shapes[n] = cur_shape;
        for (int s = 0; s < NStrides; ++s)
            tmp_strides[s][n] = cur_str[s];
        ++n;
        --d;
    }

    if (n == 0) {
        out_shapes[0] = 1;
        for (int s = 0; s < NStrides; ++s)
            out_strides[s][0] = 0;
        return 1;
    }

    // Reverse to restore outer-to-inner order; clamp writes to capacity
    // (keeps the outermost groups — callers reject clamped results anyway).
    const int out_n = n < max_out ? n : max_out;
    for (int i = 0; i < out_n; ++i) {
        int ri = n - 1 - i;
        out_shapes[i] = tmp_shapes[ri];
        for (int s = 0; s < NStrides; ++s)
            out_strides[s][i] = tmp_strides[s][ri];
    }

    return n;
}

} // namespace detail

// Squash adjacent contiguous dimensions (single stride set).
// Returns a stack-allocated result. MaxRank is the output capacity.
template <int MaxRank, typename TShape, typename TStride>
inline SquashedDims<MaxRank> squash_dimensions(
    const TShape* shapes, const TStride* strides, int rank)
{
    SquashedDims<MaxRank> overflow_guard{};
    if (rank > detail::SQUASH_MAX_INPUT_RANK) {
        // Unprocessable input rank: report a rank that never fits so the
        // caller rejects; arrays stay unwritten.
        overflow_guard.rank = rank;
        return overflow_guard;
    }
    // Promote inputs to int64_t
    int64_t s64[detail::SQUASH_MAX_INPUT_RANK];
    int64_t st64[detail::SQUASH_MAX_INPUT_RANK];
    for (int i = 0; i < rank; ++i) {
        s64[i] = static_cast<int64_t>(shapes[i]);
        st64[i] = static_cast<int64_t>(strides[i]);
    }

    SquashedDims<MaxRank> r{};
    const int64_t* strides_arr[1] = {st64};
    int64_t* out_strides_arr[1] = {r.strides};
    r.rank = detail::squash_impl<1>(s64, strides_arr, rank, MaxRank, r.shapes, out_strides_arr);
    return r;
}

// Squash adjacent contiguous dimensions (dual stride sets, e.g. contracted dims in A/B).
// Merge requires BOTH stride sets to be contiguous.
template <int MaxRank, typename TShape, typename TStride>
inline SquashedDimsDual<MaxRank> squash_dimensions_dual(
    const TShape* shapes,
    const TStride* strides_a, const TStride* strides_b,
    int rank)
{
    SquashedDimsDual<MaxRank> overflow_guard{};
    if (rank > detail::SQUASH_MAX_INPUT_RANK) {
        overflow_guard.rank = rank;
        return overflow_guard;
    }
    int64_t s64[detail::SQUASH_MAX_INPUT_RANK];
    int64_t sta64[detail::SQUASH_MAX_INPUT_RANK];
    int64_t stb64[detail::SQUASH_MAX_INPUT_RANK];
    for (int i = 0; i < rank; ++i) {
        s64[i] = static_cast<int64_t>(shapes[i]);
        sta64[i] = static_cast<int64_t>(strides_a[i]);
        stb64[i] = static_cast<int64_t>(strides_b[i]);
    }

    SquashedDimsDual<MaxRank> r{};
    const int64_t* strides_arr[2] = {sta64, stb64};
    int64_t* out_strides_arr[2] = {r.strides_a, r.strides_b};
    r.rank = detail::squash_impl<2>(s64, strides_arr, rank, MaxRank, r.shapes, out_strides_arr);
    return r;
}

} // namespace layout
} // namespace bsptc
