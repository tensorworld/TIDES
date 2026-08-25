#pragma once

// Per-tile CUTLASS grouped GEMM efficiency coefficients.
//
// Each table holds relative per-FLOP throughput for a tile shape, measured via
// bench_cutlass_tiles on the indicated GPU.  The tile selector (best_tile_for_queue)
// scores every candidate as  eff * utilization  where utilization accounts for
// M/N padding waste, then picks the highest-scoring tile.
//
// Calibration workflow:
//   1. Build and run bench_cutlass_tiles on the target GPU:
//        ./bench_cutlass_tiles --sweep                    # f32 shape sweep
//        ./bench_cutlass_tiles --dtype f64 --sweep        # f64 shape sweep
//        ./bench_cutlass_tiles                            # f32 batch-count sweep
//        ./bench_cutlass_tiles --dtype f64                # f64 batch-count sweep
//   2. Use the "Efficiency relative to 128x128 peak" output as raw coefficients.
//   3. Discount smaller tiles for CTA scheduling overhead (visible in count-sweep
//      data: smaller tiles degrade at high batch counts).
//   4. Reduce intermediate tiles (256x32, 32x256) to avoid stealing work from
//      shape-matched skinny tiles.
//   5. Update the tables below and re-run bench_pipeline_implementations to verify.

#include "kernels/large/cutlass_tile_shapes.hpp"
#include "kernels/large/shape_queue.hpp"

namespace bsptc {

struct TileCandidate { TileShape tile; int Tm, Tn; float eff; };

// ---------------------------------------------------------------------------
// cuBLAS grouped-GEMM candidate
// ---------------------------------------------------------------------------
// cuBLAS chooses its threadblock tile internally and exposes nothing to select
// or inspect, so the (Tm, Tn) padding model that gives each CUTLASS tile its
// utilization term has no counterpart: from the selector's point of view the
// candidate has utilization 1 by construction, because cuBLAS absorbs its own
// quantization.  All of the shape dependence therefore has to live in eta, and
// a single scalar will not do -- cuBLAS's efficiency varies with problem
// magnitude for exactly the reasons utilization was modelling.
//
// So eta is tabulated over min(M, N) bins (see cublas_size_bin), which is the
// same magnitude that decides which CUTLASS tile can be filled.  Both sides of
// the comparison are then on one scale: "fraction of the 128x128 tile's peak
// GFLOP/s".  The CUTLASS side is mildly optimistic on small problems (eff is a
// peak-over-shapes figure and utilization corrects only for padding), which
// biases ties toward the incumbent backend -- the safe direction.
//
// Calibration: ./bench_cutlass_tiles --cublas [--dtype f64]
struct CublasCandidate {
    float eff[NUM_CUBLAS_SIZE_BINS];  // asymptotic eta, indexed by cublas_size_bin
    // Fixed cost of one cuBLAS group, expressed as the FLOP count that would
    // take the same time at the 128x128 peak.  cuBLAS grouped GEMM charges this
    // per group regardless of how little work the group carries, which is why a
    // pure efficiency scalar mispredicts queues made of many tiny blocks.  With
    // it the score becomes
    //     1 / (1/eta + n_groups * group_cost_flops / flops_total)
    // which decays to zero as the work per group vanishes and tends to eta when
    // groups are large.  Measured by  ./bench_cutlass_tiles --cublas-groups.
    float group_cost_flops;
    bool calibrated;                  // false => never auto-selected
};// ============================================================================
// f32 SIMT — SM 8.x (calibrated on RTX A4500, SM 8.6, 46 SMs)
// ============================================================================
// 128x128 is the peak tile.  64x64 raw ≈ 1.03x but discounted for CTA overhead.
// 256x32/32x256 reduced to avoid stealing from shape-matched skinny tiles.

static constexpr TileCandidate f32_sm8x[] = {
    {TileShape::Tile_128x128, 128, 128, 1.00f},
    {TileShape::Tile_64x64,    64,  64, 0.75f},  // raw=1.03, discounted for CTA overhead
    {TileShape::Tile_32x32,    32,  32, 0.54f},  // raw=0.54
    {TileShape::Tile_256x32,  256,  32, 0.65f},  // raw=0.83, reduced
    {TileShape::Tile_32x256,   32, 256, 0.60f},  // raw=0.84, reduced
    {TileShape::Tile_128x16,  128,  16, 0.38f},  // raw=0.38
    {TileShape::Tile_16x128,   16, 128, 0.55f},  // raw=0.55
    {TileShape::Tile_64x8,     64,   8, 0.28f},  // raw=0.28
    {TileShape::Tile_8x64,      8,  64, 0.39f},  // raw=0.39
};
static constexpr int f32_sm8x_count = sizeof(f32_sm8x) / sizeof(f32_sm8x[0]);

// ============================================================================
// f32 SIMT — SM 9.x (calibrated on H100 PCIe, SM 9.0, 114 SMs)
// ============================================================================
// 64x64 is the peak tile (19-32% faster than 128x128 across all problem sizes).
// More SMs need more CTAs for full occupancy; 64x64 generates 4x more CTAs.
// N-wide tiles consistently faster than M-tall (NN layout: B is KxN row-major).

static constexpr TileCandidate f32_sm9x[] = {
    {TileShape::Tile_128x128, 128, 128, 1.00f},  // 20.4k GFLOP/s peak
    {TileShape::Tile_64x64,    64,  64, 1.15f},  // raw=1.19-1.32, discounted (gap narrows to 1.10 at high batch on large shapes)
    {TileShape::Tile_32x32,    32,  32, 0.62f},  // raw=0.67, CTA overhead discount
    {TileShape::Tile_256x32,  256,  32, 0.90f},  // raw=1.08, reduced to avoid stealing square work from 64x64
    {TileShape::Tile_32x256,   32, 256, 0.85f},  // raw=1.10, reduced similarly
    {TileShape::Tile_128x16,  128,  16, 0.42f},  // raw=0.43
    {TileShape::Tile_16x128,   16, 128, 0.65f},  // raw=0.70, slight discount
    {TileShape::Tile_64x8,     64,   8, 0.30f},  // raw=0.32
    {TileShape::Tile_8x64,      8,  64, 0.48f},  // raw=0.51
};
static constexpr int f32_sm9x_count = sizeof(f32_sm9x) / sizeof(f32_sm9x[0]);

// ============================================================================
// f64 tensor core — SM 9.x (calibrated on H100 PCIe, SM 9.0, 114 SMs)
// ============================================================================
// 128x128 DMMA 8-warp is the clear peak (25.9k–39.3k GFLOP/s).
// 32x32 TC is surprisingly strong on skinny shapes — beats dedicated skinny tiles.
// 64x8/8x64 are SIMT fallbacks (no f64 TC variant), very slow.

static constexpr TileCandidate f64_tc_sm9x[] = {
    {TileShape::Tile_128x128, 128, 128, 1.00f},  // 8 warps TC, 25.9k peak
    {TileShape::Tile_64x64,    64,  64, 0.90f},  // raw=0.95, discounted (degrades to 0.84 at batch=256 on large problems)
    {TileShape::Tile_32x32,    32,  32, 0.62f},  // raw=0.66, slight CTA discount; workhorse for skinny shapes
    {TileShape::Tile_256x32,  256,  32, 0.40f},  // raw=0.46, weak TC variant; 32x32 beats it even on 256x32 shapes
    {TileShape::Tile_32x256,   32, 256, 0.50f},  // raw=0.57, N-wide advantage over 256x32
    {TileShape::Tile_128x16,  128,  16, 0.45f},  // raw=0.47, important for very thin M-tall (1024x8)
    {TileShape::Tile_16x128,   16, 128, 0.58f},  // raw=0.63-0.66, strong TC; set below 32x32 so 32x32 wins on equal-waste
    {TileShape::Tile_64x8,     64,   8, 0.18f},  // SIMT fallback
    {TileShape::Tile_8x64,      8,  64, 0.22f},  // SIMT fallback
};
static constexpr int f64_tc_sm9x_count = sizeof(f64_tc_sm9x) / sizeof(f64_tc_sm9x[0]);

// ============================================================================
// f64 SIMT — SM 8.x (calibrated on RTX A4500, SM 8.6, 56 SMs)
// ============================================================================
// No real 128x128 (aliases to 64x64 SIMT, both measure ~419 GFLOP/s).
// f64 SIMT is throughput-bound: every tile saturates the same ~420 GFLOP/s
// ceiling on its natural large shape, so peak efficiency is ~1.0 across the
// board and tile selection is driven almost entirely by padding utilization.
// 32x32 is as fast as 64x64 here (even faster on medium-square shapes), so it
// must NOT be discounted — doing so wrongly biases away from low-waste tiles.

static constexpr TileCandidate f64_simt_sm8x[] = {
    {TileShape::Tile_64x64,    64,  64, 1.00f},  // raw=419.4, peak (aliases 128x128)
    {TileShape::Tile_32x32,    32,  32, 1.00f},  // raw=418.5, equal/faster on square
    {TileShape::Tile_256x32,  256,  32, 1.00f},  // raw=420.2
    {TileShape::Tile_32x256,   32, 256, 1.00f},  // raw=417.9
    {TileShape::Tile_128x16,  128,  16, 0.99f},  // raw=417.3
    {TileShape::Tile_16x128,   16, 128, 0.99f},  // raw=417.9
    {TileShape::Tile_64x8,     64,   8, 0.98f},  // raw=412.6
    {TileShape::Tile_8x64,      8,  64, 0.97f},  // raw=405.7
};
static constexpr int f64_simt_sm8x_count = sizeof(f64_simt_sm8x) / sizeof(f64_simt_sm8x[0]);

// ============================================================================
// cuBLAS grouped GEMM — eta by min(M,N) size bin
// ============================================================================
// Order matches cublas_size_bin: {min<16, 16-63, 64-127, 128-255, >=256}.
// Values are relative to the same 128x128 CUTLASS peak the tile coefficients
// use, so eta > 1.0 means cuBLAS beats our best square tile in that bin.
//
// Uncalibrated entries (calibrated=false) stay out of automatic selection while
// BSPTC_LARGE_BACKEND=cublas can still force the backend for A/B measurement.

// f32, SM 8.x — RTX A4500 (SM 8.6, 56 SMs), cuBLAS 13.1, 44-shape sweep at 512
// problems/shape.  eta is monotone in min(M,N) and crosses 1.0 only in the
// largest bin: cuBLAS beats the best CUTLASS tile on large aligned squares
// (1.08-1.16x measured) and loses once blocks get small, where its per-group
// overhead dominates (0.27-0.51x).  8/44 shapes won.
static constexpr CublasCandidate cublas_f32_sm8x = {
    {0.09f, 0.30f, 0.62f, 0.89f, 1.06f}, 1.7e6f, true
};

// f64, SM 8.x — same sweep.  cuBLAS won 0/44 shapes and eta peaks at 0.85, so
// the candidate never displaces a CUTLASS tile here.  f64 SIMT on SM 8.x is
// throughput-bound at ~420 GFLOP/s (see f64_simt_sm8x above): every tile
// already saturates that ceiling, leaving cuBLAS nothing to win back.
static constexpr CublasCandidate cublas_f64_sm8x = {
    {0.12f, 0.35f, 0.62f, 0.85f, 0.86f}, 0.07e6f, true
};

// SM 9.x — NOT YET MEASURED.  The FP32 gap this candidate targets is an H100
// result, so these must be calibrated on that machine before the candidate can
// be trusted there:  ./bench_cutlass_tiles --cublas [--dtype f64]
// Calibrated on H100 NVL (SM 9.0, 132 SMs), CUDA 12.6 / cuBLAS 12.6.4.
// cuBLAS beats the best CUTLASS tile on 10/44 shapes -- all of them large and
// squarish, where our SIMT FP32 tops out near 29 TFLOP/s and cuBLAS reaches
// 34-36.  That is precisely the FP32 gap this candidate exists to close.
// Bin 128-255 is noisy (128x128 1.07x, 192x192 0.69x): cuBLAS picks its own
// kernel, so shapes that are not friendly multiples fall off a cliff we cannot
// see.  The FLOP-weighted bin mean is the honest summary of that.
static constexpr CublasCandidate cublas_f32_sm9x = {
    {0.08f, 0.24f, 0.56f, 0.97f, 1.31f}, 1.71e6f, true
};
// Calibrated on H100 NVL (SM 9.0, 132 SMs), CUDA 12.6 / cuBLAS 12.6.4.
// cuBLAS loses on 0/44 shapes and peaks at eta=0.66, so it is never selected
// for f64 -- our large tier runs DMMA tensor-core kernels (128x128 at
// 33.4 TFLOP/s) while cuBLAS grouped-batched reaches only ~22.  Marked
// calibrated anyway: these are measured numbers, and the scorer declining the
// candidate on its own is the correct outcome, not a missing table.
static constexpr CublasCandidate cublas_f64_sm9x = {
    {0.08f, 0.23f, 0.38f, 0.58f, 0.66f}, 1.46e6f, true
};

} // namespace bsptc
