#pragma once

#include "pipeline_descs.hpp"
#include "gpu_sparse_input.hpp"
#include "gpu_sparse_device.cuh"
#include "preprocessing_kernel.cuh"
#include "gemm_layout_check.hpp"
#include "kernels/large/cutlass_tile_shapes.hpp"
#include "tile_efficiency.hpp"
#include "device_props.hpp"

#include <vector>
#include <cstdint>
#include <algorithm>
#include <type_traits>

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

// Default threshold for routing to CUTLASS grouped GEMM.
// Blocks with M*N >= threshold are considered "large".
constexpr int64_t DEFAULT_CUTLASS_MN_THRESHOLD = 4096;  // 64*64

// Minimum individual dimension for CUTLASS dispatch.
// Warp GEMM handles very small M or N (< 4) with max dim ≤ 1024;
// larger problems route to CUTLASS skinny tiles even with tiny min dim.
constexpr int64_t MIN_CUTLASS_DIM = 4;
constexpr int64_t MAX_WARP_DIM = 1024;

// Build a GemmProblemDesc with common fields filled in.
template <typename T>
inline GemmProblemDesc make_gemm_problem_desc(
    int32_t pair_idx, const BlockPairInfo& pair,
    const GPUSparseInput<T>& A_gpu, const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config, GemmMode gemm_mode)
{
    GemmProblemDesc desc{};
    desc.pair_idx = pair_idx;
    desc.block_idx_A = pair.block_idx_A;
    desc.block_idx_B = pair.block_idx_B;
    compute_gemm_dimensions(A_gpu, B_gpu, pair.block_idx_A, pair.block_idx_B,
                            config, desc.M, desc.N, desc.K, desc.batch_size);
    desc.lda = gemm_mode_a_transposed(gemm_mode) ? desc.M : desc.K;
    desc.ldb = gemm_mode_b_transposed(gemm_mode) ? desc.K : desc.N;
    desc.ldc = desc.N;
    desc.offset_A = A_gpu.block_offsets[pair.block_idx_A];
    desc.offset_B = B_gpu.block_offsets[pair.block_idx_B];
    return desc;
}

// Partition block pairs into three tiers per the dispatch routing design:
//   1. Warp (small):  min(M,N) < 4 AND max(M,N) ≤ 1024 — register-based, no transpose
//   2. Volume (medium): ceil(M/2)*ceil(N/2) ≤ tile_limit AND M*K+N*K ≤ vol_threshold<T>()
//      tile_limit = 256 when transpose needed (volume skips transpose),
//                   64 when no transpose (volume's only edge is persistent dispatch)
//   3. CUTLASS (large): everything else — needs transpose
// Pure classification — no device pointers, no buffer dependencies.
// Pointer arrays are built separately by build_large_pointers().
// volume_eligible is always true in normal operation; the post-squash rank guard
// in setup_medium_volume_descs handles any rank overflow.
template <typename T>
PartitionedWork<T> partition_by_threshold(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const std::vector<BlockPairInfo>& pairs,
    const ContractionConfig& config,
    GemmMode gemm_mode = GemmMode::NN,
    bool volume_eligible = true,
    bool needs_transpose = false,
    int vol_max_tiles_override = 0)
{
    PartitionedWork<T> result;
    result.all_pairs = pairs;
    result.large.gemm_mode = gemm_mode;

    // Volume shape gate: tiles ≤ 256 (8-warp CTA) when transpose is
    // needed, tiles ≤ 64 (2-warp CTA) otherwise.  Override via config.
    const int vol_tile_limit = (vol_max_tiles_override > 0)
        ? vol_max_tiles_override
        : (needs_transpose ? VOL_MAX_TILES : VOL_MAX_TILES_NO_TRANSPOSE);

    // ---- Pass 1: compute per-pair tier assignment ----
    // 0 = warp, 1 = volume, 2 = CUTLASS
    std::vector<int> per_pair_tier(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        const BlockPairInfo& pair = pairs[i];
        auto desc = make_gemm_problem_desc(
            static_cast<int32_t>(i), pair, A_gpu, B_gpu, config, gemm_mode);

        if (std::min(desc.M, desc.N) < MIN_CUTLASS_DIM &&
            std::max(desc.M, desc.N) <= MAX_WARP_DIM) {
            per_pair_tier[i] = 0;  // warp
        } else if (volume_eligible &&
                   vol_tile_count(desc.M, desc.N) <= vol_tile_limit &&
                   (static_cast<int64_t>(desc.M) * desc.K +
                    static_cast<int64_t>(desc.N) * desc.K) <= vol_threshold<T>()) {
            per_pair_tier[i] = 1;  // volume
        } else {
            per_pair_tier[i] = 2;  // CUTLASS
        }
    }

    // No promotion: volume and CUTLASS pairs can coexist for the same output
    // block.  Volume writes to d_output with atomicAdd; CUTLASS writes to
    // d_temp_buffer.  The reduce kernel (after cudaDeviceSynchronize) merges
    // temp → d_output, which is safe because all GEMM tiers are done.
    // Direct-write for CUTLASS is disabled on mixed blocks (see pipeline.cuh).

    // ---- Pass 2: build partitioned work (problems only, no pointers) ----
    for (size_t i = 0; i < pairs.size(); ++i) {
        const BlockPairInfo& pair = pairs[i];
        auto desc = make_gemm_problem_desc(
            static_cast<int32_t>(i), pair, A_gpu, B_gpu, config, gemm_mode);

        if (per_pair_tier[i] == 0) {
            result.small.pair_indices.push_back(static_cast<int64_t>(i));
            result.small.total_output_elements += pair.output_volume;
            result.small.problems.push_back(desc);
        } else if (per_pair_tier[i] == 1) {
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

// Build pointer arrays (ptr_A/B/C, lda/ldb/ldc/ldd) for CUTLASS grouped GEMM.
// Called after temp buffer is allocated and reduction plan is built.
template <typename T>
void build_large_pointers(
    LargeBlockWork<T>& large,
    const std::vector<BlockPairInfo>& pairs,
    T* d_A_values,
    T* d_B_values,
    T* d_temp_output,
    T* d_final_output,
    const int64_t* pair_dest_offsets,
    const uint8_t* pair_direct_write)
{
    if (large.empty()) return;

    large.ptr_A.clear();
    large.ptr_B.clear();
    large.ptr_C.clear();
    large.lda.clear();
    large.ldb.clear();
    large.ldc.clear();
    large.ldd.clear();

    large.ptr_A.reserve(large.num_gemm_problems);
    large.ptr_B.reserve(large.num_gemm_problems);
    large.ptr_C.reserve(large.num_gemm_problems);
    large.lda.reserve(large.num_gemm_problems);
    large.ldb.reserve(large.num_gemm_problems);
    large.ldc.reserve(large.num_gemm_problems);
    large.ldd.reserve(large.num_gemm_problems);

    for (const auto& prob : large.problems) {
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
            large.ptr_A.push_back(d_A_values + prob.offset_A + b * A_batch_stride);
            large.ptr_B.push_back(d_B_values + prob.offset_B + b * B_batch_stride);
            large.ptr_C.push_back(output_base + pair_output_offset + b * C_batch_stride);

            large.lda.push_back(prob.lda);
            large.ldb.push_back(prob.ldb);
            large.ldc.push_back(prob.ldc);
            large.ldd.push_back(prob.ldc);  // ldd same as ldc for output
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

template <typename T>
inline TileShape best_tile_for_queue(const LargeBlockWork<T>& queue_work)
{
    if (queue_work.empty()) return TileShape::Tile_128x128;

    int sm = get_sm_version_cached();

    if constexpr (std::is_same_v<T, double>) {
        if (sm >= 90)
            return score_tile_candidates(f64_tc_sm9x, f64_tc_sm9x_count, queue_work.problems, TileShape::Tile_128x128);
        else
            return score_tile_candidates(f64_simt_sm8x, f64_simt_sm8x_count, queue_work.problems, TileShape::Tile_64x64);
    } else {
        if (sm >= 90)
            return score_tile_candidates(f32_sm9x, f32_sm9x_count, queue_work.problems, TileShape::Tile_128x128);
        else
            return score_tile_candidates(f32_sm8x, f32_sm8x_count, queue_work.problems, TileShape::Tile_128x128);
    }
}

// Split the large work from PartitionedWork into per-queue structures.
// Distributes problems and aggregate counts by shape class.
// Pointer arrays (ptr_A/B/C, lda/b/c/d) are NOT copied here — call
// distribute_large_pointers after build_large_pointers to fill them.
template <typename T>
ShapeClassifiedWork<T> classify_cutlass_work(const PartitionedWork<T>& work) {
    ShapeClassifiedWork<T> classified;
    for (int i = 0; i < NUM_SHAPE_QUEUES; ++i) classified.queues[i].gemm_mode = work.large.gemm_mode;

    for (size_t j = 0; j < work.large.problems.size(); ++j) {
        const auto& prob = work.large.problems[j];
        ShapeQueue q = classify_shape(prob.M, prob.N);
        auto& queue = classified[q];

        queue.problems.push_back(prob);
        queue.source_indices.push_back(j);
        int64_t output_vol = static_cast<int64_t>(prob.M) * prob.N * prob.batch_size;
        queue.total_output_elements += output_vol;
        queue.num_gemm_problems += prob.batch_size;
    }

    return classified;
}

// Copy pointer arrays from the source large work into an already-classified
// ShapeClassifiedWork. Must be called after build_large_pointers has populated
// ptr_A/B/C and lda/b/c/d on the source.
template <typename T>
void distribute_large_pointers(ShapeClassifiedWork<T>& classified,
                               const LargeBlockWork<T>& source)
{
    // Build problem-index-to-expanded-start mapping for source
    std::vector<int64_t> expanded_start(source.problems.size());
    int64_t idx = 0;
    for (size_t j = 0; j < source.problems.size(); ++j) {
        expanded_start[j] = idx;
        idx += source.problems[j].batch_size;
    }

    for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
        auto& queue = classified.queues[qi];
        if (queue.empty()) continue;

        queue.ptr_A.clear(); queue.ptr_B.clear(); queue.ptr_C.clear();
        queue.lda.clear(); queue.ldb.clear(); queue.ldc.clear(); queue.ldd.clear();
        queue.ptr_A.reserve(queue.num_gemm_problems);
        queue.ptr_B.reserve(queue.num_gemm_problems);
        queue.ptr_C.reserve(queue.num_gemm_problems);
        queue.lda.reserve(queue.num_gemm_problems);
        queue.ldb.reserve(queue.num_gemm_problems);
        queue.ldc.reserve(queue.num_gemm_problems);
        queue.ldd.reserve(queue.num_gemm_problems);

        for (size_t i = 0; i < queue.problems.size(); ++i) {
            const auto& prob = queue.problems[i];
            size_t si = queue.source_indices[i];
            int64_t es = expanded_start[si];
            for (int64_t b = 0; b < prob.batch_size; ++b) {
                queue.ptr_A.push_back(source.ptr_A[es + b]);
                queue.ptr_B.push_back(source.ptr_B[es + b]);
                queue.ptr_C.push_back(source.ptr_C[es + b]);
                queue.lda.push_back(source.lda[es + b]);
                queue.ldb.push_back(source.ldb[es + b]);
                queue.ldc.push_back(source.ldc[es + b]);
                queue.ldd.push_back(source.ldd[es + b]);
            }
        }
    }
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

        // Merge into Square (problems only; pointers distributed later)
        auto& dst = classified[ShapeQueue::Square];
        dst.problems.insert(dst.problems.end(), src.problems.begin(), src.problems.end());
        dst.source_indices.insert(dst.source_indices.end(),
            src.source_indices.begin(), src.source_indices.end());
        dst.total_output_elements += src.total_output_elements;
        dst.num_gemm_problems += src.num_gemm_problems;

        src = LargeBlockWork<T>{};
        src.gemm_mode = dst.gemm_mode;
    }
}

} // namespace bsptc
