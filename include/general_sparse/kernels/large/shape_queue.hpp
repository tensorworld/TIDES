#pragma once

// Shape classification for Large-tier GEMM problems.
//
// Lives in its own header because two things need it: the work partitioner
// (which builds one queue per class) and the efficiency tables (the cuBLAS
// candidate's eta is binned by shape class, since cuBLAS exposes no tile for
// the utilization model to work with).

#include "../../../types.hpp"

#include <algorithm>

namespace bsptc {

enum class ShapeQueue { Square, SkinnyTall, SkinnyWide, Square32 };
constexpr int NUM_SHAPE_QUEUES = 4;

inline const char* shape_queue_name(ShapeQueue q) {
    switch (q) {
        case ShapeQueue::Square:     return "Square";
        case ShapeQueue::SkinnyTall: return "SkinnyTall";
        case ShapeQueue::SkinnyWide: return "SkinnyWide";
        case ShapeQueue::Square32:   return "Square32";
    }
    return "?";
}

// Classify a CUTLASS-tier problem by its M/N shape.
// Precondition: M >= MIN_CUTLASS_DIM && N >= MIN_CUTLASS_DIM.
// Square32: min_dim >= 16 && max_dim <= 32 (32x32 tile has strictly less padding)
// Square:   min_dim >= 16 && max_dim > 32 (64x64+ tiles)
// Skinny:   min_dim < 16, split tall/wide by M > N
inline ShapeQueue classify_shape(dim_t M, dim_t N) {
    dim_t min_dim = std::min(M, N);
    if (min_dim >= 16) {
        dim_t max_dim = std::max(M, N);
        return (max_dim <= 32) ? ShapeQueue::Square32 : ShapeQueue::Square;
    }
    return (M > N) ? ShapeQueue::SkinnyTall : ShapeQueue::SkinnyWide;
}

// ---------------------------------------------------------------------------
// Size bins for the cuBLAS candidate's eta
// ---------------------------------------------------------------------------
// A CUTLASS candidate's score is eff * utilization, where utilization supplies
// the problem-magnitude dependence analytically from the tile's (Tm, Tn).  The
// cuBLAS candidate has no tile, so the same dependence has to be tabulated, and
// min(M, N) is the quantity that drives it: it is what decides whether a large
// tile can be filled at all, and measurement shows it is also what decides
// whether cuBLAS's per-group overhead is amortised (see docs/cublas_candidate.md).
//
// Bin edges sit at the tile-selection breakpoints so the two candidate families
// change regime together.
constexpr int NUM_CUBLAS_SIZE_BINS = 5;

inline int cublas_size_bin(dim_t M, dim_t N) {
    const dim_t d = std::min(M, N);
    if (d < 16)  return 0;   // skinny: no tile fills, cuBLAS overhead dominates
    if (d < 64)  return 1;
    if (d < 128) return 2;
    if (d < 256) return 3;
    return 4;                // large square: cuBLAS is competitive here
}

inline const char* cublas_size_bin_name(int b) {
    switch (b) {
        case 0: return "min<16";
        case 1: return "16-63";
        case 2: return "64-127";
        case 3: return "128-255";
        case 4: return ">=256";
    }
    return "?";
}

} // namespace bsptc
