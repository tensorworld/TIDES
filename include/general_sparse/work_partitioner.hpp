#pragma once

#include "pipeline_descs.hpp"
#include "gpu_sparse_input.hpp"
#include "gpu_sparse_device.cuh"
#include "preprocessing_kernel.cuh"
#include "gemm_layout_check.hpp"
#include "kernels/large/cutlass_tile_shapes.hpp"
#include "kernels/large/shape_queue.hpp"
#include "kernels/large/large_backend.hpp"
#include "tile_efficiency.hpp"
#include "device_props.hpp"

#include <vector>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <cstdlib>
#include <cstring>
#include <cstdio>

// Work partitioning for dispatching block pairs to different execution paths.
// Warp (registers) / Volume (SMEM) / CUTLASS (global memory).
namespace bsptc {

// Volume kernel threshold: max M*K + N*K for medium (volume GEMM) tier.
// Float uses 8192 (32 KB SMEM); double uses 4096 (32 KB SMEM).
constexpr int64_t VOL_THRESHOLD_F32 = 8192;
constexpr int64_t VOL_THRESHOLD_F64 = 4096;

template <typename T>
constexpr int64_t vol_threshold() {
    return sizeof(T) <= 4 ? VOL_THRESHOLD_F32 : VOL_THRESHOLD_F64;
}

// Volume tile count: number of 2×2 register tiles needed for an M×N output.
inline int vol_tile_count(int M, int N) {
    return ((M + 1) / 2) * ((N + 1) / 2);
}

// Maximum tile count for volume kernel (matches 256-thread CTA).
constexpr int VOL_MAX_TILES = 256;

// Tighter tile limit when transpose is not needed.  Without transpose
// savings, only use the 64-thread (2-warp) volume kernel where all threads
// compute.  The 256-thread kernel has idle threads during compute that
// aren't justified without transpose avoidance.
constexpr int VOL_MAX_TILES_NO_TRANSPOSE = 64;

// Minimum individual dimension for CUTLASS dispatch.
// Warp GEMM handles very small M or N (< 4) with max dim ≤ 1024;
// larger problems route to CUTLASS skinny tiles even with tiny min dim.
constexpr int64_t MIN_CUTLASS_DIM = 4;
constexpr int64_t MAX_WARP_DIM = 1024;

// One entry per unique (shape_A, shape_B) pair: GEMM geometry and tier,
// computed once by build_shape_pair_plan (pipeline.cuh) and consumed per pair
// by partition_by_threshold via table lookup.
struct ShapePairInfo {
    dim_t M, N, K;
    dim_t batch_size;
    dim_t lda, ldb, ldc;
    uint8_t tier;   // 0 = warp, 1 = volume, 2 = CUTLASS
};

// Unique shape-pair plan: pair_shape_id[i] indexes shape_pairs for pair i.
struct ShapePairPlan {
    std::vector<uint32_t> pair_shape_id;   // [num_pairs]
    std::vector<ShapePairInfo> shape_pairs; // [num unique shape pairs]
};

// Partition block pairs into three tiers per the dispatch routing design:
//   1. Warp (small):  min(M,N) < 4 AND max(M,N) ≤ 1024 — register-based, no transpose
//   2. Volume (medium): ceil(M/2)*ceil(N/2) ≤ tile_limit AND M*K+N*K ≤ vol_threshold<T>()
//   3. CUTLASS (large): everything else — needs transpose
// The tier rules (including the squashed-rank eligibility that routes
// overflow shapes to CUTLASS) are evaluated once per unique shape pair in
// build_shape_pair_plan (pipeline.cuh); per-pair work here is a table lookup
// plus block-offset reads. Pure assembly — no device pointers, no buffer
// dependencies. Pointer arrays are bound per queue by bind_queue_pointers().
template <typename T>
PartitionedWork<T> partition_by_threshold(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const std::vector<BlockPairInfo>& pairs,
    const ShapePairPlan& sp,
    GemmMode gemm_mode)
{
    PartitionedWork<T> result;
    result.large.gemm_mode = gemm_mode;

    // ---- Pass 1: tier counts so pass 2 can reserve exactly ----
    size_t tier_counts[3] = {0, 0, 0};
    for (size_t i = 0; i < pairs.size(); ++i)
        tier_counts[sp.shape_pairs[sp.pair_shape_id[i]].tier]++;

    // No promotion: volume and CUTLASS pairs can coexist for the same output
    // block.  Volume writes to d_output with atomicAdd; CUTLASS writes to
    // d_temp_buffer.  The reduce kernel (after cudaDeviceSynchronize) merges
    // temp → d_output, which is safe because all GEMM tiers are done.
    // Direct-write for CUTLASS is disabled on mixed blocks (see pipeline.cuh).

    // ---- Pass 2: build partitioned work (problems only, no pointers) ----
    result.small.pair_indices.reserve(tier_counts[0]);
    result.small.problems.reserve(tier_counts[0]);
    result.medium.pair_indices.reserve(tier_counts[1]);
    result.medium.problems.reserve(tier_counts[1]);
    result.large.problems.reserve(tier_counts[2]);
    for (size_t i = 0; i < pairs.size(); ++i) {
        const BlockPairInfo& pair = pairs[i];
        const uint32_t sid = sp.pair_shape_id[i];
        const ShapePairInfo& info = sp.shape_pairs[sid];

        GemmProblemDesc desc{};
        desc.pair_idx = static_cast<int32_t>(i);
        desc.block_idx_A = pair.block_idx_A;
        desc.block_idx_B = pair.block_idx_B;
        desc.M = info.M;
        desc.N = info.N;
        desc.K = info.K;
        desc.batch_size = info.batch_size;
        desc.lda = info.lda;
        desc.ldb = info.ldb;
        desc.ldc = info.ldc;
        desc.offset_A = A_gpu.block_offsets[pair.block_idx_A];
        desc.offset_B = B_gpu.block_offsets[pair.block_idx_B];
        desc.shape_pair_id = sid;

        if (info.tier == 0) {
            result.small.pair_indices.push_back(static_cast<int64_t>(i));
            result.small.total_output_elements += pair.output_volume;
            result.small.problems.push_back(desc);
        } else if (info.tier == 1) {
            result.medium.pair_indices.push_back(static_cast<int64_t>(i));
            result.medium.total_output_elements += pair.output_volume;
            result.medium.problems.push_back(desc);
        } else {
            result.large.problems.push_back(desc);

            int64_t output_volume = static_cast<int64_t>(desc.M) * desc.N * desc.batch_size;
            result.large.total_output_elements += output_volume;
            result.large.num_gemm_problems += desc.batch_size;
        }
    }

    return result;
}

// Bind batch-expanded pointer arrays (ptr_A/B/C) directly onto one shape
// queue's work. Called after the temp buffer is allocated and the reduction
// plan is built. Leading dimensions are not materialized — the CUTLASS
// arg-staging loop generates them from the descs.
template <typename T>
void bind_queue_pointers(
    LargeBlockWork<T>& queue,
    const std::vector<BlockPairInfo>& pairs,
    T* d_A_values,
    T* d_B_values,
    T* d_temp_output,
    T* d_final_output,
    const int64_t* pair_dest_offsets,
    const uint8_t* pair_direct_write)
{
    if (queue.empty()) return;

    queue.ptr_A.clear();
    queue.ptr_B.clear();
    queue.ptr_C.clear();
    queue.ptr_A.reserve(queue.num_gemm_problems);
    queue.ptr_B.reserve(queue.num_gemm_problems);
    queue.ptr_C.reserve(queue.num_gemm_problems);

    for (const auto& prob : queue.problems) {
        const bool write_direct_to_final =
            (pair_direct_write[prob.pair_idx] != 0);
        const int64_t pair_output_offset = write_direct_to_final
            ? pair_dest_offsets[prob.pair_idx]
            : pairs[prob.pair_idx].output_offset;
        T* output_base = write_direct_to_final ? d_final_output : d_temp_output;
        int64_t A_batch_stride = static_cast<int64_t>(prob.M) * prob.K;
        int64_t B_batch_stride = static_cast<int64_t>(prob.K) * prob.N;
        int64_t C_batch_stride = static_cast<int64_t>(prob.M) * prob.N;

        for (int64_t b = 0; b < prob.batch_size; ++b) {
            queue.ptr_A.push_back(d_A_values + prob.offset_A + b * A_batch_stride);
            queue.ptr_B.push_back(d_B_values + prob.offset_B + b * B_batch_stride);
            queue.ptr_C.push_back(output_base + pair_output_offset + b * C_batch_stride);
        }
    }
}

// Get statistics about the partitioning for debugging/logging
template <typename T>
struct PartitionStats {
    int64_t num_large_pairs;
    int64_t num_medium_pairs;
    int64_t num_small_pairs;
    int64_t num_gemm_problems;
    int64_t large_output_elements;
    int64_t medium_output_elements;
    int64_t small_output_elements;
    int64_t total_output_elements;
    int64_t min_mn_large;
    int64_t max_mn_large;
    int64_t min_mn_small;
    int64_t max_mn_small;
};

template <typename T>
PartitionStats<T> compute_partition_stats(const PartitionedWork<T>& work) {
    PartitionStats<T> stats{};

    stats.num_large_pairs = static_cast<int64_t>(work.large.problems.size());
    stats.num_medium_pairs = static_cast<int64_t>(work.medium.pair_indices.size());
    stats.num_small_pairs = static_cast<int64_t>(work.small.pair_indices.size());
    stats.num_gemm_problems = work.large.num_gemm_problems;
    stats.large_output_elements = work.large.total_output_elements;
    stats.medium_output_elements = work.medium.total_output_elements;
    stats.small_output_elements = work.small.total_output_elements;
    stats.total_output_elements = stats.large_output_elements + stats.medium_output_elements + stats.small_output_elements;

    stats.min_mn_large = INT64_MAX;
    stats.max_mn_large = 0;
    stats.min_mn_small = INT64_MAX;
    stats.max_mn_small = 0;

    for (const auto& prob : work.large.problems) {
        int64_t mn = static_cast<int64_t>(prob.M) * prob.N;
        stats.min_mn_large = std::min(stats.min_mn_large, mn);
        stats.max_mn_large = std::max(stats.max_mn_large, mn);
    }

    if (stats.min_mn_large == INT64_MAX) stats.min_mn_large = 0;
    if (stats.min_mn_small == INT64_MAX) stats.min_mn_small = 0;

    return stats;
}

// Print three-tier shape diagnostic for all block pairs.
// Classification matches partition_by_threshold exactly:
//   1. Warp (small):  min(M,N) < 4 AND max(M,N) <= 1024
//   2. Volume (medium): ceil(M/2)*ceil(N/2) <= 256 AND M*K+N*K <= vol_threshold<T>()
//   3. CUTLASS (large): everything else
template <typename T>
void print_tier_diagnostic(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const std::vector<BlockPairInfo>& pairs,
    const ContractionConfig& config)
{
    struct TierStats {
        int64_t count = 0;
        int64_t total_output = 0;
        int64_t total_flops = 0;
        int64_t min_M = INT64_MAX, max_M = 0;
        int64_t min_N = INT64_MAX, max_N = 0;
        int64_t min_K = INT64_MAX, max_K = 0;
        int64_t min_MN = INT64_MAX, max_MN = 0;
        int64_t M_buckets[5] = {};
        int64_t N_buckets[5] = {};

        void add(int64_t M, int64_t N, int64_t K, int64_t batch) {
            count++;
            total_output += M * N * batch;
            total_flops += 2 * M * N * K * batch;
            min_M = std::min(min_M, M); max_M = std::max(max_M, M);
            min_N = std::min(min_N, N); max_N = std::max(max_N, N);
            min_K = std::min(min_K, K); max_K = std::max(max_K, K);
            int64_t mn = M * N;
            min_MN = std::min(min_MN, mn); max_MN = std::max(max_MN, mn);
            auto bucket = [](int64_t v) -> int {
                if (v < 8)  return 0;
                if (v < 16) return 1;
                if (v < 32) return 2;
                if (v < 64) return 3;
                return 4;
            };
            M_buckets[bucket(M)]++;
            N_buckets[bucket(N)]++;
        }
    };

    TierStats large, medium, small;

    for (const auto& pair : pairs) {
        dim_t M, N, K, batch;
        compute_gemm_dimensions(A_gpu, B_gpu, pair.block_idx_A, pair.block_idx_B,
                                config, M, N, K, batch);

        // Match partition_by_threshold order: warp → volume → CUTLASS
        if (std::min(M, N) < MIN_CUTLASS_DIM &&
            std::max(M, N) <= MAX_WARP_DIM) {
            small.add(M, N, K, batch);
        } else if (vol_tile_count(M, N) <= VOL_MAX_TILES &&
                   (static_cast<int64_t>(M) * K +
                    static_cast<int64_t>(N) * K) <= vol_threshold<T>()) {
            medium.add(M, N, K, batch);
        } else {
            large.add(M, N, K, batch);
        }
    }

    auto print_tier = [](const char* name, const TierStats& s) {
        if (s.count == 0) {
            printf("  %-8s  %6ld pairs\n", name, s.count);
            return;
        }
        printf("  %-8s  %6ld pairs  %10ld output_elems  %12ld MFlops\n",
               name, s.count, s.total_output, s.total_flops / 1000000);
        printf("           M=[%ld,%ld]  N=[%ld,%ld]  K=[%ld,%ld]  MN=[%ld,%ld]\n",
               s.min_M, s.max_M, s.min_N, s.max_N, s.min_K, s.max_K,
               s.min_MN, s.max_MN);
        const char* bucket_labels[] = {"[1,7]", "[8,15]", "[16,31]", "[32,63]", "[64+]"};
        printf("           M dist:");
        for (int i = 0; i < 5; i++) if (s.M_buckets[i]) printf(" %s=%ld", bucket_labels[i], s.M_buckets[i]);
        printf("\n           N dist:");
        for (int i = 0; i < 5; i++) if (s.N_buckets[i]) printf(" %s=%ld", bucket_labels[i], s.N_buckets[i]);
        printf("\n");
    };

    printf("  Three-tier shape diagnostic (%ld pairs, vol_threshold=%ld):\n",
           (long)pairs.size(), (long)vol_threshold<T>());
    print_tier("Large", large);
    print_tier("Medium", medium);
    print_tier("Small", small);
}

// ============================================================================
// Shape classification for CUTLASS-tier problems
// ============================================================================

// ShapeQueue / NUM_SHAPE_QUEUES / classify_shape now live in
// kernels/large/shape_queue.hpp, shared with the efficiency tables.

// Per-queue work for shape-classified CUTLASS dispatch.
template <typename T>
struct ShapeClassifiedWork {
    LargeBlockWork<T> queues[NUM_SHAPE_QUEUES];  // indexed by ShapeQueue

    LargeBlockWork<T>& operator[](ShapeQueue q) { return queues[static_cast<int>(q)]; }
    const LargeBlockWork<T>& operator[](ShapeQueue q) const { return queues[static_cast<int>(q)]; }

    int num_nonempty() const {
        int n = 0;
        for (int i = 0; i < NUM_SHAPE_QUEUES; ++i) if (!queues[i].empty()) ++n;
        return n;
    }
};

// Original fixed tile per shape queue (superseded by best_tile_for_queue).
inline TileShape tile_for_queue(ShapeQueue queue)
{
    switch (queue) {
        case ShapeQueue::SkinnyTall:  return TileShape::Tile_64x8;
        case ShapeQueue::SkinnyWide:  return TileShape::Tile_8x64;
        case ShapeQueue::Square:      return TileShape::Tile_128x128;
        case ShapeQueue::Square32:    return TileShape::Tile_32x32;
    }
    return TileShape::Tile_128x128;
}

// Cached SM version for tile selection (same pattern as gemm_dispatch.cu).
inline int get_sm_version_cached() {
    static int sm = -1;
    if (sm < 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        sm = props.major * 10 + props.minor;
    }
    return sm;
}

// Adaptive tile selection: pick the tile that maximizes eff * utilization
// across all problems in the queue.  Efficiency coefficients are defined in
// tile_efficiency.hpp (per-GPU calibration via bench_cutlass_tiles).
//
// Scoring: for each candidate, compute padding waste across all problems,
// then score = eff * (useful_elements / padded_elements).  Highest score wins.

inline TileShape score_tile_candidates(
    const TileCandidate* cands, int n_cands,
    const std::vector<GemmProblemDesc>& problems,
    TileShape fallback)
{
    float best_score = -1.0f;
    TileShape best = fallback;
    for (int ci = 0; ci < n_cands; ++ci) {
        const auto& c = cands[ci];
        int64_t total_useful = 0;
        int64_t total_padded = 0;
        for (const auto& prob : problems) {
            int64_t padM = static_cast<int64_t>((prob.M + c.Tm - 1) / c.Tm) * c.Tm;
            int64_t padN = static_cast<int64_t>((prob.N + c.Tn - 1) / c.Tn) * c.Tn;
            int64_t K = prob.K;
            total_useful += static_cast<int64_t>(prob.M) * prob.N * K * prob.batch_size;
            total_padded += padM * padN * K * prob.batch_size;
        }
        float util = total_padded > 0 ? float(total_useful) / float(total_padded) : 0.0f;
        float score = c.eff * util;
        if (score > best_score) {
            best_score = score;
            best = c.tile;
        }
    }
    return best;
}

// Best-scoring CUTLASS tile plus the score it achieved, so the cuBLAS
// candidate can be ranked against it on the same scale.
inline float score_of_tile(const TileCandidate* cands, int n_cands,
                           const std::vector<GemmProblemDesc>& problems,
                           TileShape tile)
{
    for (int ci = 0; ci < n_cands; ++ci) {
        if (cands[ci].tile != tile) continue;
        const auto& c = cands[ci];
        int64_t total_useful = 0, total_padded = 0;
        for (const auto& prob : problems) {
            int64_t padM = static_cast<int64_t>((prob.M + c.Tm - 1) / c.Tm) * c.Tm;
            int64_t padN = static_cast<int64_t>((prob.N + c.Tn - 1) / c.Tn) * c.Tn;
            total_useful += static_cast<int64_t>(prob.M) * prob.N * prob.K * prob.batch_size;
            total_padded += padM * padN * prob.K * prob.batch_size;
        }
        float util = total_padded > 0 ? float(total_useful) / float(total_padded) : 0.0f;
        return c.eff * util;
    }
    return 0.0f;
}

// cuBLAS score.  Two terms, because cuBLAS's cost has two parts:
//
//   throughput -- utilization is 1 by construction (no user-visible tile to pad
//                 against), so the throughput term is eta alone, FLOP-weighted
//                 across the size bins present in the queue;
//   per group  -- a fixed cost charged once per cuBLAS group (one group per
//                 GemmProblemDesc), which is what makes cuBLAS collapse on
//                 queues of many tiny blocks even though its per-FLOP rate on
//                 those shapes looks merely mediocre.
//
// Combining them as times and renormalising back to the "fraction of 128x128
// peak" scale the tile scores live on:
//
//   score = flops / (flops/eta + n_groups * c) = 1 / (1/eta + n_groups*c/flops)
//
inline float score_cublas_candidate(const CublasCandidate& cand,
                                    const std::vector<GemmProblemDesc>& problems)
{
    if (!cand.calibrated) return -1.0f;
    double weighted = 0.0, total = 0.0;
    for (const auto& prob : problems) {
        const double flops = static_cast<double>(prob.M) * prob.N *
                             prob.K * prob.batch_size;
        weighted += flops * static_cast<double>(cand.eff[cublas_size_bin(prob.M, prob.N)]);
        total += flops;
    }
    if (total <= 0.0) return -1.0f;
    const double eta = weighted / total;
    if (eta <= 0.0) return -1.0f;
    const double overhead = static_cast<double>(problems.size()) *
                            static_cast<double>(cand.group_cost_flops) / total;
    return static_cast<float>(1.0 / (1.0 / eta + overhead));
}

// Backend forced by BSPTC_LARGE_BACKEND=cutlass|cublas|auto (default auto).
// Exists so the two backends can be A/B'd on identical plans without rebuilding.
enum class BackendOverride { Auto, ForceCutlass, ForceCublas };

inline BackendOverride large_backend_override() {
    static BackendOverride mode = [] {
        const char* e = std::getenv("BSPTC_LARGE_BACKEND");
        if (!e) return BackendOverride::Auto;
        if (std::strcmp(e, "cublas") == 0)  return BackendOverride::ForceCublas;
        if (std::strcmp(e, "cutlass") == 0) return BackendOverride::ForceCutlass;
        return BackendOverride::Auto;
    }();
    return mode;
}

// Select the Large-tier kernel for a queue: the highest-scoring candidate
// across both backends.  CUTLASS candidates score eff * utilization; the cuBLAS
// candidate scores eta alone (see score_cublas_candidate).
template <typename T>
inline KernelChoice best_kernel_for_queue(const LargeBlockWork<T>& queue_work)
{
    if (queue_work.empty()) return KernelChoice(TileShape::Tile_128x128);

    const int sm = get_sm_version_cached();
    const TileCandidate* cands;
    int n_cands;
    TileShape fallback;
    const CublasCandidate* cublas;

    if constexpr (std::is_same_v<T, double>) {
        if (sm >= 90) {
            cands = f64_tc_sm9x; n_cands = f64_tc_sm9x_count;
            fallback = TileShape::Tile_128x128; cublas = &cublas_f64_sm9x;
        } else {
            cands = f64_simt_sm8x; n_cands = f64_simt_sm8x_count;
            fallback = TileShape::Tile_64x64; cublas = &cublas_f64_sm8x;
        }
    } else {
        if (sm >= 90) {
            cands = f32_sm9x; n_cands = f32_sm9x_count;
            fallback = TileShape::Tile_128x128; cublas = &cublas_f32_sm9x;
        } else {
            cands = f32_sm8x; n_cands = f32_sm8x_count;
            fallback = TileShape::Tile_128x128; cublas = &cublas_f32_sm8x;
        }
    }

    const TileShape best_tile =
        score_tile_candidates(cands, n_cands, queue_work.problems, fallback);

    switch (large_backend_override()) {
        case BackendOverride::ForceCutlass:
            return KernelChoice(LargeBackend::CUTLASS, best_tile);
        case BackendOverride::ForceCublas:
            return KernelChoice(LargeBackend::CUBLAS, best_tile);
        case BackendOverride::Auto:
            break;
    }

    const float tile_score =
        score_of_tile(cands, n_cands, queue_work.problems, best_tile);
    const float cublas_score = score_cublas_candidate(*cublas, queue_work.problems);

    if (std::getenv("BSPTC_DUMP_KERNEL_CHOICE")) {
        double wf = 0, tot = 0, wmin = 0;
        int64_t nexp = 0;
        for (const auto& p : queue_work.problems) {
            double f = double(p.M) * p.N * p.K * p.batch_size;
            tot += f; wmin += f * std::min(p.M, p.N); wf += f / std::max<int64_t>(p.batch_size,1);
            nexp += p.batch_size;
        }
        std::fprintf(stderr,
            "[choice] groups=%zu expanded=%lld flops=%.3gG wmean_min(M,N)=%.1f "
            "flops/prob=%.3gM | tile=%s score=%.3f | cublas score=%.3f -> %s\n",
            queue_work.problems.size(), (long long)nexp, tot / 1e9,
            tot > 0 ? wmin / tot : 0.0, tot / std::max<size_t>(queue_work.problems.size(),1) / 1e6,
            tile_shape_name(best_tile), double(tile_score), double(cublas_score),
            cublas_score > tile_score ? "CUBLAS" : "CUTLASS");
    }

    if (cublas_score > tile_score)
        return KernelChoice(LargeBackend::CUBLAS, best_tile);
    return KernelChoice(LargeBackend::CUTLASS, best_tile);
}

// Backwards-compatible tile-only selector (used by the tile calibration bench).
template <typename T>
inline TileShape best_tile_for_queue(const LargeBlockWork<T>& queue_work)
{
    return best_kernel_for_queue(queue_work).tile;
}

// Split the large work from PartitionedWork into per-queue structures.
// Distributes problems and aggregate counts by shape class.
// Pointer arrays (ptr_A/B/C) are NOT filled here — call bind_queue_pointers
// per queue once the output/temp buffers exist.
template <typename T>
ShapeClassifiedWork<T> classify_cutlass_work(const PartitionedWork<T>& work) {
    ShapeClassifiedWork<T> classified;
    for (int i = 0; i < NUM_SHAPE_QUEUES; ++i) classified.queues[i].gemm_mode = work.large.gemm_mode;

    for (size_t j = 0; j < work.large.problems.size(); ++j) {
        const auto& prob = work.large.problems[j];
        ShapeQueue q = classify_shape(prob.M, prob.N);
        auto& queue = classified[q];

        queue.problems.push_back(prob);
        int64_t output_vol = static_cast<int64_t>(prob.M) * prob.N * prob.batch_size;
        queue.total_output_elements += output_vol;
        queue.num_gemm_problems += prob.batch_size;
    }

    return classified;
}

// Merge skinny queues into Square when their native-tile CTA count is small.
// A queue whose total CTAs (M,N tiled by its native tile × batch_size) fits
// within a few SM waves is launch-overhead-dominated — the tile choice barely
// matters, so merging avoids the inter-queue sync cost.
template <typename T>
void merge_small_queues(ShapeClassifiedWork<T>& classified)
{
    const int64_t cta_threshold = 8LL * get_num_sms();

    for (int qi = 1; qi < NUM_SHAPE_QUEUES; ++qi) {
        auto& src = classified.queues[qi];
        if (src.empty()) continue;

        TileShape native = tile_for_queue(static_cast<ShapeQueue>(qi));
        int tM = tile_m(native);
        int tN = tile_n(native);

        int64_t total_ctas = 0;
        for (const auto& prob : src.problems) {
            total_ctas += static_cast<int64_t>((prob.M + tM - 1) / tM)
                        * ((prob.N + tN - 1) / tN) * prob.batch_size;
        }

        if (total_ctas >= cta_threshold) continue;

        // Merge into Square (problems only; pointers bound later)
        auto& dst = classified[ShapeQueue::Square];
        dst.problems.insert(dst.problems.end(), src.problems.begin(), src.problems.end());
        dst.total_output_elements += src.total_output_elements;
        dst.num_gemm_problems += src.num_gemm_problems;

        src = LargeBlockWork<T>{};
        src.gemm_mode = dst.gemm_mode;
    }
}

} // namespace bsptc
