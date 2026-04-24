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

namespace bsptc {

struct TileCandidate { TileShape tile; int Tm, Tn; float eff; };

// ============================================================================
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
// f64 SIMT — SM 8.x (estimated, not yet calibrated via bench_cutlass_tiles)
// ============================================================================
// No real 128x128 (aliases to 64x64 SIMT).  64x64 8-warp SIMT is the best.

static constexpr TileCandidate f64_simt_sm8x[] = {
    {TileShape::Tile_64x64,    64,  64, 0.90f},  // 8 warps SIMT (best available)
    {TileShape::Tile_32x32,    32,  32, 0.40f},  // 4 warps SIMT
    {TileShape::Tile_256x32,  256,  32, 0.35f},  // 8 warps SIMT intermediate
    {TileShape::Tile_32x256,   32, 256, 0.35f},
    {TileShape::Tile_128x16,  128,  16, 0.30f},  // 4 warps SIMT intermediate
    {TileShape::Tile_16x128,   16, 128, 0.30f},
    {TileShape::Tile_64x8,     64,   8, 0.20f},  // 2 warps SIMT
    {TileShape::Tile_8x64,      8,  64, 0.20f},
};
static constexpr int f64_simt_sm8x_count = sizeof(f64_simt_sm8x) / sizeof(f64_simt_sm8x[0]);

} // namespace bsptc
