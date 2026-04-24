#pragma once

#include "pipeline_descs.hpp"
#include "gpu_sparse_input.hpp"
#include "gpu_sparse_device.cuh"
#include "preprocessing_kernel.cuh"
#include "work_partitioner.hpp"
#include "kernels/medium/v2_volume_gemm.cuh"
#include "kernels/small/warp_gemm.cuh"
#include "analysis/pipeline.hpp"
#include "analysis/dispatchers.hpp"
#include "analysis/passes/batch_expand.hpp"
#include "block_shape_map.hpp"
#include "hash_tables.hpp"
#include "../einsum.hpp"
#include "gpu_accumulator.cuh"
#include "transpose_kernel.cuh"
#include "gemm_layout_check.hpp"
#include "kernels/transpose/nd_hypertile.cuh"
#include "kernels/transpose/hypertile_index_builder.hpp"
#include "../layout_utils.hpp"

#include "batch_partitioner.hpp"
#include "cutlass_grouped_gemm.hpp"
#include "kernels/large/cutlass_tile_shapes.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <numeric>

namespace bsptc {

// ============================================================================
// Shared types (used by pipeline and backward-compat executor)
// ============================================================================

// Result from general sparse contraction
template <typename T>
struct GeneralSparseResult {
    T* d_output = nullptr;
    bool owns_output = true;  // false when user provided d_output
    std::vector<int64_t> output_offsets;
    std::vector<int64_t> output_volumes;
    int64_t total_output_elements = 0;
    int64_t num_output_blocks = 0;
    std::vector<int32_t> output_block_coords;
    dim_t output_rank = 0;

    GeneralSparseResult() = default;
    ~GeneralSparseResult() { if (owns_output && d_output) cudaFree(d_output); }
    GeneralSparseResult(GeneralSparseResult&& o) noexcept
        : d_output(o.d_output), owns_output(o.owns_output),
          output_offsets(std::move(o.output_offsets)),
          output_volumes(std::move(o.output_volumes)),
          total_output_elements(o.total_output_elements),
          num_output_blocks(o.num_output_blocks),
          output_block_coords(std::move(o.output_block_coords)),
          output_rank(o.output_rank)
    { o.d_output = nullptr; }
    GeneralSparseResult& operator=(GeneralSparseResult&& o) noexcept {
        if (this != &o) {
            if (owns_output && d_output) cudaFree(d_output);
            d_output = o.d_output; o.d_output = nullptr;
            owns_output = o.owns_output;
            output_offsets = std::move(o.output_offsets);
            output_volumes = std::move(o.output_volumes);
            total_output_elements = o.total_output_elements;
            num_output_blocks = o.num_output_blocks;
            output_block_coords = std::move(o.output_block_coords);
            output_rank = o.output_rank;
        }
        return *this;
    }
    GeneralSparseResult(const GeneralSparseResult&) = delete;
    GeneralSparseResult& operator=(const GeneralSparseResult&) = delete;
};

// Timing breakdown for pipeline stages (all in milliseconds).
struct PipelineTimingBreakdown {
    double preprocess_ms = 0;
    double accum_map_ms = 0;
    double transpose_ms = 0;
    double contraction_ms = 0;
    double accumulation_ms = 0;
    double total_ms = 0;
    // Phase-level timing (plan + prepare + allocate = setup, execute = total_ms)
    double plan_ms = 0;
    double prepare_ms = 0;      // Address-independent shape/geometry build (pipeline_prepare)
    double allocate_ms = 0;     // GPU buffer binding, pointer patching, upload (pipeline_allocate)
    double bind_ms = 0;         // binding/upload subset of allocate (excludes malloc/memset/free)
    double bind_upload_ms = 0;   // upload_accumulation_map + upload_reduction_plan
    double bind_patch_ms = 0;    // pointer patching + CUTLASS arg prep + small/medium binding
    double bind_stream_ms = 0;   // streaming/transpose setup
    int small_descs = 0;
    int small_shapes = 0;       // unique shapes (shape/mem split)
    int small_warps = 0;        // total warps dispatched
    int large_gemm_problems = 0;
    std::string large_tile_info;   // e.g. "Square=64x64" for non-streaming
    int medium_descs = 0;
    int medium_shapes = 0;      // unique shapes (shape/mem split)
    int64_t large_flops = 0;
    int64_t medium_flops = 0;
    int64_t small_flops = 0;
    int64_t large_output_elems = 0;
    int64_t medium_output_elems = 0;
    int64_t small_output_elems = 0;
    int64_t transpose_bytes = 0;
    const char* transpose_A_kind = nullptr;
    const char* transpose_B_kind = nullptr;
    const char* small_kernel_kind = nullptr;
    // Transpose dispatch detail (perm string + key dim info)
    std::string transpose_A_detail;
    std::string transpose_B_detail;
    // Estimated peak GPU memory (bytes) — all pipeline-owned allocations at peak
    int64_t peak_gpu_bytes = 0;
    // Streaming info (populated when streaming is enabled)
    bool streaming_enabled = false;
    int streaming_total_batches = 0;
    int64_t streaming_buf_cap = 0;
    struct QueueInfo {
        int batches = 0;
        int problems = 0;
        const char* tile_name = nullptr;
    };
    QueueInfo streaming_queues[NUM_SHAPE_QUEUES];
    int64_t streaming_precomp_bytes = 0;  // pre-allocated transpose descriptor buffers (total)
};

namespace detail_timing {
inline double ms_since(const std::chrono::high_resolution_clock::time_point& t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
} // namespace detail_timing

// Build ContractionConfig from EinsumIndexSets and tensor labels
inline ContractionConfig build_contraction_config(
    const EinsumIndexSets& index_sets,
    const std::vector<char>& A_labels,
    const std::vector<char>& B_labels,
    const std::string& output_labels = "")
{
    ContractionConfig config;
    config.num_contracted = 0;
    config.num_free_A = 0;
    config.num_free_B = 0;
    config.num_batch = 0;

    auto find_pos = [](const std::vector<char>& labels, char c) -> int32_t {
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == c) return static_cast<int32_t>(i);
        }
        return -1;
    };

    auto find_pos_str = [](const std::string& labels, char c) -> int32_t {
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == c) return static_cast<int32_t>(i);
        }
        return -1;
    };

    std::vector<char> internal_order;

    for (char c : index_sets.batch_indices) {
        int32_t pos_A = find_pos(A_labels, c);
        int32_t pos_B = find_pos(B_labels, c);
        if (pos_A >= 0 && pos_B >= 0) {
            config.batch_pos_A[config.num_batch] = pos_A;
            config.batch_pos_B[config.num_batch] = pos_B;
            config.num_batch++;
            internal_order.push_back(c);
        }
    }

    for (char c : index_sets.free_indices_A) {
        int32_t pos_A = find_pos(A_labels, c);
        if (pos_A >= 0) {
            config.free_pos_A[config.num_free_A] = pos_A;
            config.num_free_A++;
            internal_order.push_back(c);
        }
    }

    for (char c : index_sets.free_indices_B) {
        int32_t pos_B = find_pos(B_labels, c);
        if (pos_B >= 0) {
            config.free_pos_B[config.num_free_B] = pos_B;
            config.num_free_B++;
            internal_order.push_back(c);
        }
    }

    for (char c : index_sets.contracted_indices) {
        int32_t pos_A = find_pos(A_labels, c);
        int32_t pos_B = find_pos(B_labels, c);
        if (pos_A >= 0 && pos_B >= 0) {
            config.contracted_pos_A[config.num_contracted] = pos_A;
            config.contracted_pos_B[config.num_contracted] = pos_B;
            config.num_contracted++;
        }
    }

    config.output_rank = static_cast<int32_t>(internal_order.size());

    if (!output_labels.empty()) {
        for (int32_t i = 0; i < config.output_rank; ++i) {
            char label = internal_order[i];
            int32_t out_pos = find_pos_str(output_labels, label);
            config.internal_to_output[i] = out_pos;
        }
    } else {
        for (int32_t i = 0; i < config.output_rank; ++i) {
            config.internal_to_output[i] = i;
        }
    }

    return config;
}

// Hypertile transpose kernel configuration — shared between task builder and launcher.
constexpr int HT_TILE_VOL = 1024;   // tile volume (elements per tile)
constexpr int HT_BLOCK_SIZE = 256;  // CUDA threads per block

struct DispatchConfig {
    int64_t mn_threshold = DEFAULT_CUTLASS_MN_THRESHOLD;
    int64_t transpose_buf_cap = 0; // 0 = disabled (legacy full-size workspace); >0 = streamed transpose buffer cap in elements
    float output_beta = 0;  // 0 = zero d_output before contraction; 1 = accumulate onto existing d_output
    int vol_max_tiles = 0;  // 0 = auto (256 w/ transpose, 64 w/o); >0 = override
};

// Workspace requirements returned by pipeline_query_workspace.
// The user allocates these buffers and passes them to pipeline_allocate.
struct PipelineWorkspaceInfo {
    int64_t output_bytes = 0;     // d_output size (always needed)
    int64_t workspace_bytes = 0;  // temp buffer + transpose buffers (may be 0)
    int64_t output_elements = 0;  // convenience: output_bytes / sizeof(T)
};

template <typename T>
struct TransposedValues {
    T* d_A;
    T* d_B;
    T* d_workspace_A;
    T* d_workspace_B;
    GemmMode gemm_mode;
};


// Precomputed metadata for batched hypertile transpose.
// Built once during pipeline_setup, reused during pipeline_execute.
struct TransposeFlatMeta {
    TransposeDispatchKind kind = TransposeDispatchKind::None;
    bool     valid = false;

    // Hypertile generic permute
    // Per-rank GPU slices (shapes + mems, grouped by squashed_rank for templated kernel)
    struct HTSlice {
        void*    d_shapes = nullptr;       // HypertileShape<R>[]
        void*    d_mems = nullptr;         // HypertileMem[]
        int32_t* d_prefix_sum = nullptr;
        int32_t  num_tasks = 0;
        int32_t  num_shapes = 0;
        int32_t  total_tiles = 0;
        int32_t  rank = 0;
    };
    HTSlice  ht_slices[HT_MAX_RANK];      // indexed by rank-1 (only populated ranks used)
    int32_t  ht_num_slices = 0;
    int32_t  ht_total_tiles = 0;           // sum across all slices
    int32_t  ht_squashed_rank = 0;         // for diagnostics (max across tasks)
    int32_t  ht_tile_volume = 0;           // for diagnostics (first task)
    int32_t  ht_min_rr = 0;               // min read-squashed rank
    int32_t  ht_min_wr = 0;               // min write-squashed rank

    bool owns_gpu = true;  // false when using pre-allocated buffers (streaming)

    void free_gpu() {
        if (!owns_gpu) { valid = false; return; }
        for (int i = 0; i < ht_num_slices; ++i) {
            if (ht_slices[i].d_shapes) { cudaFree(ht_slices[i].d_shapes); ht_slices[i].d_shapes = nullptr; }
            if (ht_slices[i].d_mems) { cudaFree(ht_slices[i].d_mems); ht_slices[i].d_mems = nullptr; }
            if (ht_slices[i].d_prefix_sum) { cudaFree(ht_slices[i].d_prefix_sum); ht_slices[i].d_prefix_sum = nullptr; }
        }
        valid = false;
    }
};

// Pre-allocated GPU buffers for streaming transpose precomp (one per slot per operand).
// Eliminates cudaMalloc inside the streaming batch loop.
struct TransposePrecompBuffers {
    void* d_shapes = nullptr;           // HypertileShape<HT_MAX_RANK>[] (loose upper bound)
    void* d_mems = nullptr;             // HypertileMem[]
    int32_t* d_prefix_sum = nullptr;
    size_t shapes_bytes = 0;
    size_t mems_bytes = 0;
    size_t prefix_sum_bytes = 0;

    size_t total_bytes() const {
        return shapes_bytes + mems_bytes + prefix_sum_bytes;
    }

    void allocate(int num_blocks) {
        // Shapes: worst case one per block, use max-rank size
        shapes_bytes = static_cast<size_t>(num_blocks) * sizeof(HypertileShape<HT_MAX_RANK>);
        check_cuda(cudaMalloc(&d_shapes, shapes_bytes), "precomp_buf: ht shapes");
        // Mems: one per block
        mems_bytes = static_cast<size_t>(num_blocks) * sizeof(HypertileMem);
        check_cuda(cudaMalloc(&d_mems, mems_bytes), "precomp_buf: ht mems");
        // Prefix sum: each rank-group slice contributes (num_tasks + 1) entries;
        // worst case all blocks are non-trivial across HT_MAX_RANK slices.
        prefix_sum_bytes = static_cast<size_t>(num_blocks + HT_MAX_RANK) * sizeof(int32_t);
        check_cuda(cudaMalloc(&d_prefix_sum, prefix_sum_bytes), "precomp_buf: ht prefix_sum");
    }

    void free_gpu() {
        if (d_shapes) { cudaFree(d_shapes); d_shapes = nullptr; }
        if (d_mems) { cudaFree(d_mems); d_mems = nullptr; }
        if (d_prefix_sum) { cudaFree(d_prefix_sum); d_prefix_sum = nullptr; }
    }
};

// Precompute per-shape trivial transpose flags for one operand.
// Reads the pre-computed trivial flag from HypertileIndex.
inline std::vector<uint8_t> compute_trivial_shapes(
    const BlockShapeMap& smap,
    const HypertileIndex* ht_index = nullptr)
{
    std::vector<uint8_t> trivial(smap.num_unique, 0);

    if (ht_index && ht_index->valid) {
        for (int s = 0; s < smap.num_unique; ++s)
            trivial[s] = ht_index->per_shape[s].trivial ? 1 : 0;
    }

    return trivial;
}

// Assemble per-batch hypertile transpose metadata using the hypertile index.
// Only iterates the given block_indices, assembles lightweight HypertileMem
// and prefix_sum. When bufs != nullptr, uses pre-allocated GPU buffers (streaming).
// When bufs == nullptr, self-allocates GPU memory (non-streaming).
template <typename T>
TransposeFlatMeta assemble_hypertile_batch(
    const std::vector<int>& block_indices,
    const GPUSparseInput<T>& gpu,
    const BlockShapeMap& shape_map,
    const HypertileIndex& cache,
    const std::vector<uint8_t>& trivial_shapes,
    const std::vector<int64_t>& pack_offset = {},
    TransposePrecompBuffers* bufs = nullptr,
    cudaStream_t stream = 0)
{
    TransposeFlatMeta pc;
    pc.kind = TransposeDispatchKind::GenericPermute;
    if (block_indices.empty()) return pc;

    // Per-slice accumulators (mems only; prefix_sum built after sort)
    struct SliceAccum {
        std::vector<HypertileMem> mems;
        std::vector<int32_t> prefix_sum;
        int32_t total_tiles = 0;
    };
    std::vector<SliceAccum> accums(cache.num_slices);

    // Iterate block_indices, look up cache entry, build mems
    for (size_t bi = 0; bi < block_indices.size(); ++bi) {
        int b = block_indices[bi];
        if (gpu.block_volumes[b] == 0) continue;

        uint32_t shape_id = shape_map.block_to_shape[b];
        if (trivial_shapes[shape_id])
            continue;
        const auto& entry = cache.per_shape[shape_id];

        HypertileMem mem{};
        mem.src_base = gpu.block_offsets[b];
        mem.dst_base = pack_offset.empty() ? gpu.block_offsets[b] : pack_offset[bi];
        mem.shape_idx = entry.shape_idx;

        auto& acc = accums[entry.slice_idx];
        acc.mems.push_back(mem);
        acc.total_tiles += entry.num_tiles;
    }

    // Sort mems by shape_idx for register-map reuse, then build prefix_sum
    for (int s = 0; s < cache.num_slices; ++s) {
        auto& acc = accums[s];
        if (acc.mems.empty()) continue;

        // Counting sort by shape_idx (small number of unique shapes)
        if (acc.mems.size() > 1) {
            int max_shape = cache.slices[s].num_shapes;
            std::vector<int> counts(max_shape, 0);
            for (const auto& m : acc.mems) counts[m.shape_idx]++;

            std::vector<int> offsets(max_shape, 0);
            for (int i = 1; i < max_shape; ++i) offsets[i] = offsets[i-1] + counts[i-1];

            std::vector<HypertileMem> sorted(acc.mems.size());
            for (const auto& m : acc.mems) {
                sorted[offsets[m.shape_idx]++] = m;
            }
            acc.mems = std::move(sorted);
        }

        // Build prefix_sum from sorted mems using the side array.
        const auto& si = cache.slices[s];
        acc.prefix_sum.push_back(0);
        for (const auto& m : acc.mems) {
            acc.prefix_sum.push_back(acc.prefix_sum.back() + si.h_num_tiles[m.shape_idx]);
        }
    }

    // Fill TransposeFlatMeta, using pre-allocated buffers (streaming) or self-allocating
    pc.owns_gpu = (bufs == nullptr);
    int32_t total_tiles = 0;
    size_t shapes_offset = 0, mems_offset = 0, ps_offset = 0;
    int slice_count = 0;

    for (int s = 0; s < cache.num_slices; ++s) {
        const auto& acc = accums[s];
        if (acc.mems.empty()) continue;

        auto& dst = pc.ht_slices[slice_count];
        dst.rank = cache.slices[s].rank;
        dst.num_tasks = static_cast<int32_t>(acc.mems.size());
        dst.num_shapes = cache.slices[s].num_shapes;
        dst.total_tiles = acc.total_tiles;
        total_tiles += acc.total_tiles;

        size_t sb = cache.slices[s].shape_bytes;
        size_t mb = acc.mems.size() * sizeof(HypertileMem);
        size_t ps = acc.prefix_sum.size() * sizeof(int32_t);

        if (bufs) {
            if (shapes_offset + sb > bufs->shapes_bytes)
                throw std::runtime_error("assemble_hypertile_batch: shapes overflow");
            if (mems_offset + mb > bufs->mems_bytes)
                throw std::runtime_error("assemble_hypertile_batch: mems overflow");
            if (ps_offset + ps > bufs->prefix_sum_bytes)
                throw std::runtime_error("assemble_hypertile_batch: prefix_sum overflow");
            dst.d_shapes = static_cast<char*>(bufs->d_shapes) + shapes_offset;
            dst.d_mems = static_cast<char*>(bufs->d_mems) + mems_offset;
            dst.d_prefix_sum = reinterpret_cast<int32_t*>(
                reinterpret_cast<char*>(bufs->d_prefix_sum) + ps_offset);
        } else {
            check_cuda(cudaMalloc(&dst.d_shapes, sb), "assemble: malloc shapes");
            check_cuda(cudaMalloc(&dst.d_mems, mb), "assemble: malloc mems");
            check_cuda(cudaMalloc(&dst.d_prefix_sum, ps), "assemble: malloc prefix_sum");
        }

        check_cuda(cudaMemcpyAsync(dst.d_shapes, cache.slices[s].h_shape_bytes.data(), sb,
                                   cudaMemcpyHostToDevice, stream), "assemble: cpy shapes");
        check_cuda(cudaMemcpyAsync(dst.d_mems, acc.mems.data(), mb,
                                   cudaMemcpyHostToDevice, stream), "assemble: cpy mems");
        check_cuda(cudaMemcpyAsync(dst.d_prefix_sum, acc.prefix_sum.data(), ps,
                                   cudaMemcpyHostToDevice, stream), "assemble: cpy prefix_sum");

        shapes_offset += sb;
        mems_offset += mb;
        ps_offset += ps;
        slice_count++;
    }

    pc.ht_num_slices = slice_count;
    pc.ht_total_tiles = total_tiles;
    pc.ht_squashed_rank = cache.max_squashed_rank;
    pc.ht_tile_volume = cache.max_tile_volume;
    pc.ht_min_rr = cache.min_rr;
    pc.ht_min_wr = cache.min_wr;
    pc.valid = (slice_count > 0);
    return pc;
}

// Launch precomputed batched hypertile transpose — one kernel per rank group.
template <typename T>
void launch_batched_transpose(
    const T* d_src, T* d_dst,
    const TransposeFlatMeta& meta,
    cudaStream_t tiled_stream = 0)
{
    if (!meta.valid) return;
    for (int s = 0; s < meta.ht_num_slices; ++s) {
        const auto& sl = meta.ht_slices[s];
        if (sl.num_tasks == 0) continue;
        launch_hypertile_kernel<T, HT_TILE_VOL, HT_BLOCK_SIZE>(
            d_src, d_dst, sl.d_shapes, sl.d_mems, sl.d_prefix_sum,
            sl.num_tasks, sl.total_tiles,
            sl.rank, tiled_stream);
    }
}

template <typename T>
TransposedValues<T> ensure_gemm_layout(
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    const GemmLayoutPlan& layout_plan,
    const TransposeFlatMeta& precomp_A,
    const TransposeFlatMeta& precomp_B,
    T* d_workspace_A,
    T* d_workspace_B,
    cudaStream_t tiled_stream = 0)
{
    TransposedValues<T> tv;
    tv.d_A = A_dev.d_values;
    tv.d_B = B_dev.d_values;
    tv.d_workspace_A = d_workspace_A;
    tv.d_workspace_B = d_workspace_B;
    tv.gemm_mode = GemmMode::NN;

    if (layout_plan.plan_A.transpose && precomp_A.kind != TransposeDispatchKind::None) {
        launch_batched_transpose<T>(A_dev.d_values, tv.d_workspace_A, precomp_A, tiled_stream);
        tv.d_A = tv.d_workspace_A;
    }

    if (layout_plan.plan_B.transpose && precomp_B.kind != TransposeDispatchKind::None) {
        launch_batched_transpose<T>(B_dev.d_values, tv.d_workspace_B, precomp_B, tiled_stream);
        tv.d_B = tv.d_workspace_B;
    }

    tv.gemm_mode = layout_plan.mode_after_transpose;
    return tv;
}

template <typename T>
void free_transposed_values(TransposedValues<T>& tv) {
    if (tv.d_workspace_A) { cudaFree(tv.d_workspace_A); tv.d_workspace_A = nullptr; }
    if (tv.d_workspace_B) { cudaFree(tv.d_workspace_B); tv.d_workspace_B = nullptr; }
}

// ============================================================================
// M-dimension splitting for small/medium kernels
// ============================================================================

// When M >= this threshold, split the sub-problem in half along M.
// Reduces per-CTA work and improves occupancy for tall-M blocks.

// ============================================================================
// Tier state: GPU-side dispatch data for one kernel tier (small or medium)
// ============================================================================

// ============================================================================
// PipelinePlan: all state between setup and execute
// ============================================================================

template <typename T>
struct PipelinePlan {
    // Core config
    ContractionConfig config{};
    dim_t output_rank = 0;

    // Preprocessing (host-side hash join, no GPU allocs)
    int64_t num_pairs = 0;
    std::vector<BlockPairInfo> pairs_host;
    int64_t total_pair_output_size = 0;
    AccumulationMap accum_map{};

    // Layout & transpose
    GemmLayoutPlan layout_plan{};
    GemmMode gemm_mode = GemmMode::NN;
    bool needs_transpose = false;
    TransposedValues<T> tv{};
    TransposeFlatMeta transpose_precomp_A;
    TransposeFlatMeta transpose_precomp_B;

    // Block shape dedup (per-tensor, built in pipeline_plan)
    BlockShapeMap shape_map_A;
    BlockShapeMap shape_map_B;

    // Hypertile transpose index (per-operand, built in pipeline_prepare)
    HypertileIndex hypertile_index_A;
    HypertileIndex hypertile_index_B;

    // Per-shape trivial transpose flags (indexed by shape_id, built in pipeline_prepare)
    std::vector<uint8_t> trivial_shapes_A;
    std::vector<uint8_t> trivial_shapes_B;

    // Work partitioning
    PartitionedWork<T> work{};

    // Large tier
    ReductionPlan reduce_plan{};

    // Non-streaming per-queue CUTLASS dispatch
    ShapeClassifiedWork<T> nonstream_classified;
    CutlassGemmRunnerResult nonstream_runners[NUM_SHAPE_QUEUES];

    // Small tier (warp kernel) — split shape/mem
    void* d_small_shapes = nullptr;          // type-erased WarpShape array (few unique shapes)
    WarpMem* d_small_mems = nullptr;         // per-desc memory (A/B/C ptrs + shape_idx)
    int small_num_descs = 0;
    int small_num_shapes = 0;
    int32_t* d_warp_prefix = nullptr;    // prefix sum of warps per desc (num_descs+1)
    int total_warps = 0;
    int warp_variant = 0;                    // 4-bit: (INT_LARGE<<3)|(INT_K<<2)|(CKA<<1)|CKB

    // Medium tier (volume GEMM — scatter-gather load + 2×2 register-tile, bypasses transpose)
    VolumeShape* d_medium_vol_shapes = nullptr;
    VolumeMem* d_medium_vol_mems = nullptr;
    int medium_vol_num_shapes = 0;
    int medium_vol_num_problems = 0;
    int medium_vol_max_smem_elems = 0;
    int medium_vol_max_tiles = 0;  // max ceil(M/2)*ceil(N/2) for kernel selection

    // Buffers
    T* d_temp_buffer = nullptr;
    T* d_output = nullptr;

    // Workspace ownership (false when user provides buffers via pipeline_allocate)
    bool owns_output = true;
    bool owns_workspace = true;   // governs d_temp_buffer + streaming d_buf
    float output_beta = 0;        // saved from DispatchConfig for phase 2

    // Staging from pipeline_plan → pipeline_allocate
    ShapeClassifiedWork<T> staging_classified;

    // Prepared state (pipeline_prepare → pipeline_allocate)
    struct PreparedSmall {
        std::vector<uint8_t> h_shape_bytes;  // type-erased WarpShape array
        struct PerShape {
            int64_t batch_stride_a, batch_stride_b, batch_stride_c;
            int32_t warps_per;
        };
        std::vector<PerShape> shape_strides;
        std::vector<uint32_t> key_indices;
        std::vector<int32_t> h_warp_prefix;
        int total_descs = 0;
        int num_shapes = 0;
        int total_warps = 0;
        bool contig_k_a = false;  // set by prepare_small_shapes_impl
        bool contig_k_b = false;
    };
    PreparedSmall prepared_small;

    struct PreparedMedium {
        std::vector<VolumeShape> h_shapes;
        struct PerShape {
            int64_t batch_stride_a, batch_stride_b, batch_stride_c;
        };
        std::vector<PerShape> shape_strides;
        std::vector<uint32_t> key_indices;
        int total_descs = 0;
        int num_shapes = 0;
        int max_smem_elems = 0;
        int max_tiles = 0;
        // Problems grouped by output block (CSR: offsets + flat indices).
        // Built in pipeline_prepare; used by bind_medium_mems to emit mems
        // in C-pointer order without an O(n log n) sort.
        std::vector<int32_t> ob_offsets;   // [num_active_obs + 1]
        std::vector<int32_t> ob_prob_idx;  // [n_probs] problem indices grouped by OB
        std::vector<int32_t> ob_batch_sizes; // [n_probs] batch_size per grouped problem
    };
    PreparedMedium prepared_medium;

    // Non-streaming tile shapes (from pipeline_prepare)
    TileShape nonstream_tiles[NUM_SHAPE_QUEUES] = {};

    // Streams
    cudaStream_t stream_large = nullptr;
    cudaStream_t stream_small = nullptr;
    cudaStream_t stream_medium = nullptr;

    // Streamed large-tier transpose (bounded memory TTGT)
    struct StreamingState {
        // Shared across queues (sequential execution)
        T* d_buf[2] = {};            // double transpose buffers
        CutlassGemmRunnerResult args[2];  // pre-allocated CUTLASS arg buffers
        cudaStream_t streams[2] = {};     // two alternating streams
        cudaEvent_t gemm_done[2] = {};    // GEMM completion events
        cudaEvent_t copy_done[2] = {};    // H2D args copy completion (for h_args_buf safety)
        int64_t buf_cap = 0;         // elements per buffer
        int64_t max_expanded_problems = 0;
        bool enabled = false;

        // Pre-allocated transpose descriptor buffers (2 slots × 2 operands)
        TransposePrecompBuffers precomp_bufs[2][2];  // [slot][operand: 0=A, 1=B]

        // Per-queue (indexed by ShapeQueue)
        BatchPartitionResult queue_partitions[NUM_SHAPE_QUEUES];
        LargeBlockWork<T> queue_work[NUM_SHAPE_QUEUES];  // final work with ptr_C
        TileShape queue_tiles[NUM_SHAPE_QUEUES];

        // Precomputed GEMM + transpose state (first ≤2 batches across all queues).
        // Uses args[0]/args[1] for GEMM, precomp_bufs[slot][op] for transpose.
        // At execute time, only transpose kernels + GEMM launch remain.
        bool precomputed = false;
        int num_precomp = 0;
        TransposeFlatMeta precomp_meta[2][2];  // [batch_slot][operand: 0=A, 1=B]
    };
    StreamingState streaming;

    // Cross-stream sync events for transpose → GEMM ordering
    cudaEvent_t transpose_done_flat = nullptr;
    cudaEvent_t transpose_done_tiled = nullptr;

    // Timing
    PipelineTimingBreakdown* timing = nullptr;

    // Measured pipeline allocation (bytes) via cudaMemGetInfo delta.
    // Set in pipeline_allocate; used by populate_timing_stats for accurate peak.
    int64_t measured_alloc_bytes = 0;

    PipelinePlan() = default;
    ~PipelinePlan() {
        // Big buffers: only free if we own them (not user-provided workspace)
        if (owns_workspace) {
            cudaFree(streaming.d_buf[0]);
            cudaFree(streaming.d_buf[1]);
            cudaFree(d_temp_buffer);
            free_transposed_values(tv);
        } else {
            tv.d_workspace_A = nullptr;
            tv.d_workspace_B = nullptr;
        }
        streaming.args[0].free_buffers();
        streaming.args[1].free_buffers();
        for (int s = 0; s < 2; ++s)
            for (int op = 0; op < 2; ++op)
                streaming.precomp_bufs[s][op].free_gpu();
        if (streaming.streams[0]) cudaStreamDestroy(streaming.streams[0]);
        if (streaming.streams[1]) cudaStreamDestroy(streaming.streams[1]);
        if (streaming.gemm_done[0]) cudaEventDestroy(streaming.gemm_done[0]);
        if (streaming.gemm_done[1]) cudaEventDestroy(streaming.gemm_done[1]);
        if (streaming.copy_done[0]) cudaEventDestroy(streaming.copy_done[0]);
        if (streaming.copy_done[1]) cudaEventDestroy(streaming.copy_done[1]);
        cudaFree(d_small_shapes);
        cudaFree(d_small_mems);
        cudaFree(d_warp_prefix);
        cudaFree(d_medium_vol_shapes);
        cudaFree(d_medium_vol_mems);
        free_reduction_plan(reduce_plan);
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi)
            nonstream_runners[qi].free_buffers();
        transpose_precomp_A.free_gpu();
        transpose_precomp_B.free_gpu();
        free_accumulation_map(accum_map);
        if (stream_large) cudaStreamDestroy(stream_large);
        if (stream_small) cudaStreamDestroy(stream_small);
        if (stream_medium) cudaStreamDestroy(stream_medium);
        if (transpose_done_flat) cudaEventDestroy(transpose_done_flat);
        if (transpose_done_tiled) cudaEventDestroy(transpose_done_tiled);
    }
    // Non-movable, non-copyable. Use pipeline_setup(plan, ...) output parameter.
    PipelinePlan(PipelinePlan&&) = delete;
    PipelinePlan& operator=(PipelinePlan&&) = delete;
    PipelinePlan(const PipelinePlan&) = delete;
    PipelinePlan& operator=(const PipelinePlan&) = delete;
};

// ============================================================================
// Timing/diagnostic stats (depends on PipelinePlan, included after its definition)
#include "pipeline_stats.hpp"

// ============================================================================
// pipeline_setup / pipeline_execute / pipeline_extract_result
// ============================================================================

// Check if batch dims occupy leading positions 0..num_batch-1.
inline bool batch_dims_are_leading(const int32_t* batch_pos, int32_t num_batch) {
    for (int32_t i = 0; i < num_batch; ++i)
        if (batch_pos[i] != i) return false;
    return true;
}

// Compute stride for dimension at position pos in a row-major block.
// stride = product of dims[pos+1..rank-1]
inline dim_t compute_dim_stride(const dim_t* block_dims, dim_t rank, int32_t pos) {
    dim_t stride = 1;
    for (int32_t j = pos + 1; j < rank; ++j)
        stride *= block_dims[j];
    return stride;
}

// Shape + batch strides for a unique pair shape key.
// Geometry filled from block dims; batch strides used by bind_small_mems.
template <typename DescType>
struct SmallShapeTemplate {
    using ShapeType = WarpShape<typename DescType::LargeFreeType,
                                typename DescType::ContractedType>;
    ShapeType shape;
    int64_t batch_stride_a;
    int64_t batch_stride_b;
    int64_t batch_stride_c;
};

// Compute a single effective stride for a free dim set whose product is small (≤ 3).
// After squash_dimensions, any product ≤ 3 reduces to at most one non-trivial dim.
inline dim_t compute_small_dim_stride(const dim_t* block_dims, dim_t rank,
                                      const int32_t* free_pos, int32_t num_free) {
    if (num_free == 1)
        return compute_dim_stride(block_dims, rank, free_pos[0]);
    // Multiple free dims with product ≤ 3 → squash; find the one non-trivial dim.
    int32_t f_shapes[MAX_RANK]; int64_t f_strides[MAX_RANK];
    for (int32_t fi = 0; fi < num_free; ++fi) {
        int32_t pos = free_pos[fi];
        f_shapes[fi] = static_cast<int32_t>(block_dims[pos]);
        f_strides[fi] = compute_dim_stride(block_dims, rank, pos);
    }
    auto sq = layout::squash_dimensions<WARP_MAX_DIMS>(f_shapes, f_strides, num_free);
    for (int i = 0; i < sq.rank; ++i)
        if (sq.shapes[i] > 1) return static_cast<dim_t>(sq.strides[i]);
    return 0;  // product == 1, stride doesn't matter
}

// Build warp shape geometry for a specific desc type (address-independent).
// Dedup pair shape keys, build WarpShape directly, compute warp prefix.
// Stores result in plan.prepared_small for later bind_small_mems.
template <typename DescType, typename T>
void prepare_small_shapes_impl(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    using ShapeType = typename SmallShapeTemplate<DescType>::ShapeType;
    using LargeFreeType = typename DescType::LargeFreeType;

    const auto& problems = plan.work.small.problems;
    const int n_probs = static_cast<int>(problems.size());
    if (n_probs == 0) return;

    const auto& config = plan.config;
    const dim_t A_rank = A_gpu.rank;
    const dim_t B_rank = B_gpu.rank;

    // Phase 1: dedup pair shape keys.
    std::vector<uint64_t> pair_keys(n_probs);
    for (int i = 0; i < n_probs; ++i)
        pair_keys[i] = make_pair_shape_key(
            plan.shape_map_A.block_to_shape[problems[i].block_idx_A],
            plan.shape_map_B.block_to_shape[problems[i].block_idx_B]);

    std::vector<uint32_t> key_indices(n_probs);
    std::vector<int> first_occ(n_probs);
    int num_unique = pair_key_dedup(pair_keys.data(), n_probs,
                                    key_indices.data(), first_occ.data());

    // Build shape + stride template per unique pair shape key.
    // Also track K contiguity: if ANY shape has squashed innermost K
    // stride != 1, the contiguous fast path is unsafe.
    bool all_contig_k_a = true, all_contig_k_b = true;
    using Tmpl = SmallShapeTemplate<DescType>;
    std::vector<Tmpl> templates(num_unique);
    for (int u = 0; u < num_unique; ++u) {
        const auto& prob = problems[first_occ[u]];
        const dim_t* A_dims = plan.shape_map_A.shape_dims(
            plan.shape_map_A.block_to_shape[prob.block_idx_A]);
        const dim_t* B_dims = plan.shape_map_B.shape_dims(
            plan.shape_map_B.block_to_shape[prob.block_idx_B]);

        Tmpl& tmpl = templates[u];
        tmpl = Tmpl{};
        auto& s = tmpl.shape;

        dim_t M = 1, N = 1, K = 1;
        for (int32_t i = 0; i < config.num_free_A; ++i)
            M *= A_dims[config.free_pos_A[i]];
        for (int32_t i = 0; i < config.num_free_B; ++i)
            N *= B_dims[config.free_pos_B[i]];
        for (int32_t i = 0; i < config.num_contracted; ++i)
            K *= A_dims[config.contracted_pos_A[i]];

        // Canonical small/large assignment: smaller free dim gets bare stride.
        const bool m_small = (M <= N);
        s.small_size = m_small ? M : N;
        s.large_size = m_small ? N : M;
        s.K_total = K;
        s.m_is_small = m_small ? 1 : 0;

        tmpl.batch_stride_a = 1;
        tmpl.batch_stride_b = 1;
        for (int32_t d = config.num_batch; d < A_rank; ++d)
            tmpl.batch_stride_a *= A_dims[d];
        for (int32_t d = config.num_batch; d < B_rank; ++d)
            tmpl.batch_stride_b *= B_dims[d];
        tmpl.batch_stride_c = static_cast<int64_t>(M) * N;

        // --- Small dim stride (always a bare stride) ---
        if (m_small) {
            // Small dim = M (A's free dims)
            s.small_stride = compute_small_dim_stride(
                A_dims, A_rank, config.free_pos_A, config.num_free_A);
        } else {
            // Small dim = N (B's free dims)
            s.small_stride = compute_small_dim_stride(
                B_dims, B_rank, config.free_pos_B, config.num_free_B);
        }

        // --- Large free dim (StdDim or IntDim) ---
        // When m_small: large = N → B's free dims.
        // When !m_small: large = M → A's free dims.
        const dim_t* large_dims = m_small ? B_dims : A_dims;
        dim_t large_rank = m_small ? B_rank : A_rank;
        const int32_t* large_pos = m_small ? config.free_pos_B : config.free_pos_A;
        int32_t num_large = m_small ? config.num_free_B : config.num_free_A;

        if constexpr (std::is_same_v<LargeFreeType, IntDim>) {
            if (num_large > 1) {
                int32_t f_shapes[MAX_RANK]; int64_t f_strides[MAX_RANK];
                for (int32_t fi = 0; fi < num_large; ++fi) {
                    int32_t pos = large_pos[fi];
                    f_shapes[fi] = static_cast<int32_t>(large_dims[pos]);
                    f_strides[fi] = compute_dim_stride(large_dims, large_rank, pos);
                }
                auto sq = layout::squash_dimensions<WARP_MAX_DIMS>(
                    f_shapes, f_strides, num_large);
                s.large.num_dims = static_cast<int8_t>(sq.rank);
                for (int i = 0; i < sq.rank; ++i) {
                    s.large.dims[i] = static_cast<dim_t>(sq.shapes[i]);
                    s.large.strides[i] = static_cast<dim_t>(sq.strides[i]);
                }
            } else {
                // Single free dim but INT_LARGE selected conservatively.
                // Fill IntDim with num_dims=1.
                s.large.num_dims = 1;
                s.large.dims[0] = s.large_size;
                if (num_large > 0)
                    s.large.strides[0] = compute_dim_stride(
                        large_dims, large_rank, large_pos[0]);
            }
        } else {
            if (num_large > 0)
                s.large.stride = compute_dim_stride(
                    large_dims, large_rank, large_pos[0]);
        }

        // Fill k (contracted dims)
        if constexpr (std::is_same_v<decltype(s.k), IntK>) {
            int32_t k_shapes[MAX_RANK];
            int64_t k_strides_a[MAX_RANK], k_strides_b[MAX_RANK];
            for (int32_t ci = 0; ci < config.num_contracted; ++ci) {
                int32_t pos_a = config.contracted_pos_A[ci];
                int32_t pos_b = config.contracted_pos_B[ci];
                k_shapes[ci] = static_cast<int32_t>(A_dims[pos_a]);
                k_strides_a[ci] = compute_dim_stride(A_dims, A_rank, pos_a);
                k_strides_b[ci] = compute_dim_stride(B_dims, B_rank, pos_b);
            }
            auto sq = layout::squash_dimensions_dual<WARP_MAX_DIMS>(
                k_shapes, k_strides_a, k_strides_b, config.num_contracted);
            s.k.num_dims = static_cast<int8_t>(sq.rank);
            for (int i = 0; i < sq.rank; ++i) {
                s.k.dims[i] = static_cast<dim_t>(sq.shapes[i]);
                s.k.strides_a[i] = static_cast<dim_t>(sq.strides_a[i]);
                s.k.strides_b[i] = static_cast<dim_t>(sq.strides_b[i]);
            }
            s.k.k_start = 0;
            // Track contiguity from squashed strides
            if (sq.strides_a[sq.rank - 1] != 1) all_contig_k_a = false;
            if (sq.strides_b[sq.rank - 1] != 1) all_contig_k_b = false;
        } else {
            if (config.num_contracted > 0) {
                s.k.stride_a = compute_dim_stride(A_dims, A_rank, config.contracted_pos_A[0]);
                s.k.stride_b = compute_dim_stride(B_dims, B_rank, config.contracted_pos_B[0]);
                if (s.k.stride_a != 1) all_contig_k_a = false;
                if (s.k.stride_b != 1) all_contig_k_b = false;
            }
        }
    }

    // Store shapes directly in prepared state (no intermediate copy).
    auto& ps = plan.prepared_small;
    ps.contig_k_a = all_contig_k_a;
    ps.contig_k_b = all_contig_k_b;
    ps.num_shapes = num_unique;
    ps.h_shape_bytes.resize(num_unique * sizeof(ShapeType));
    for (int u = 0; u < num_unique; ++u)
        std::memcpy(ps.h_shape_bytes.data() + u * sizeof(ShapeType),
                    &templates[u].shape, sizeof(ShapeType));

    // Per-shape strides (address-independent, used by bind_small_mems).
    ps.shape_strides.resize(num_unique);
    for (int u = 0; u < num_unique; ++u) {
        ps.shape_strides[u].batch_stride_a = templates[u].batch_stride_a;
        ps.shape_strides[u].batch_stride_b = templates[u].batch_stride_b;
        ps.shape_strides[u].batch_stride_c = templates[u].batch_stride_c;
        ps.shape_strides[u].warps_per =
            (templates[u].shape.small_size * templates[u].shape.large_size + 31) / 32;
    }
    ps.key_indices = std::move(key_indices);

    // Count total expanded descs and build warp prefix (address-independent).
    int total_descs = 0;
    for (int p = 0; p < n_probs; ++p)
        total_descs += problems[p].batch_size;
    ps.total_descs = total_descs;

    if (total_descs > 0) {
        ps.h_warp_prefix.resize(total_descs + 1);
        ps.h_warp_prefix[0] = 0;
        int idx = 0;
        for (int p = 0; p < n_probs; ++p) {
            int32_t warps = ps.shape_strides[ps.key_indices[p]].warps_per;
            for (dim_t b = 0; b < problems[p].batch_size; ++b, ++idx)
                ps.h_warp_prefix[idx + 1] = ps.h_warp_prefix[idx] + warps;
        }
        ps.total_warps = ps.h_warp_prefix[total_descs];
    }

    plan.small_num_descs = total_descs;
    plan.small_num_shapes = num_unique;
    plan.total_warps = ps.total_warps;
}

// Build warp shape geometry (address-independent, no GPU alloc).
// Dispatches to prepare_small_shapes_impl based on dim interleaving.
// INT_LARGE = true if either operand has multi-dim free dims (conservative:
// the large dim might sometimes be the simple one, in which case IntDim
// is filled with num_dims=1 — a trivial single-multiply decode).
template <typename T>
void prepare_small_shapes(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    const auto& config = plan.config;
    bool int_large = (config.num_free_A > 1 || config.num_free_B > 1);
    bool int_k = (config.num_contracted > 1);

    // Build shapes first — contiguity is determined from squashed strides
    // inside prepare_small_shapes_impl, not from original dim positions.
    int desc_type = (int_large ? 2 : 0) | (int_k ? 1 : 0);
    switch (desc_type) {
        case 0: prepare_small_shapes_impl<DescSS>(plan, A_gpu, B_gpu); break;
        case 1: prepare_small_shapes_impl<DescSI>(plan, A_gpu, B_gpu); break;
        case 2: prepare_small_shapes_impl<DescIS>(plan, A_gpu, B_gpu); break;
        case 3: prepare_small_shapes_impl<DescII>(plan, A_gpu, B_gpu); break;
    }

    // Use squash-derived contiguity from prepared state.
    // The original position-based check can be wrong when squash eliminates
    // size-1 contracted dims, changing which dim is innermost.
    bool contig_k_a = plan.prepared_small.contig_k_a;
    bool contig_k_b = plan.prepared_small.contig_k_b;

    plan.warp_variant = (int_large ? 8 : 0) | (int_k ? 4 : 0) |
                        (contig_k_a ? 2 : 0) | (contig_k_b ? 1 : 0);
}

// Fill WarpMem with absolute GPU pointers and upload shapes+mems+prefix to GPU.
// Uses prepared state from prepare_small_shapes.
template <typename T>
void bind_small_mems(
    PipelinePlan<T>& plan,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev)
{
    const auto& ps = plan.prepared_small;
    if (ps.total_descs == 0) return;

    const auto& problems = plan.work.small.problems;
    const int n_probs = static_cast<int>(problems.size());

    std::vector<WarpMem> h_mems(ps.total_descs);

    int idx = 0;
    for (int p = 0; p < n_probs; ++p) {
        const auto& prob = problems[p];
        const auto& ss = ps.shape_strides[ps.key_indices[p]];
        int64_t ob = plan.accum_map.h_pair_output_block_idx[prob.pair_idx];
        int8_t atomic = (plan.accum_map.h_ob_tier_flags[ob] & kOBMultiSmall) ? 1 : 0;

        const T* base_A = A_dev.d_values + prob.offset_A;
        const T* base_B = B_dev.d_values + prob.offset_B;
        T* base_C = plan.d_output + plan.accum_map.h_dest_offsets[prob.pair_idx];

        for (dim_t b = 0; b < prob.batch_size; ++b, ++idx) {
            h_mems[idx].A = base_A + b * ss.batch_stride_a;
            h_mems[idx].B = base_B + b * ss.batch_stride_b;
            h_mems[idx].C = base_C + b * ss.batch_stride_c;
            h_mems[idx].shape_idx = ps.key_indices[p];
            h_mems[idx].use_atomic = atomic;
        }
    }

    // Upload shapes
    check_cuda(cudaMalloc(&plan.d_small_shapes, ps.h_shape_bytes.size()),
               "malloc warp_shapes");
    check_cuda(cudaMemcpyAsync(plan.d_small_shapes, ps.h_shape_bytes.data(),
                               ps.h_shape_bytes.size(),
                               cudaMemcpyHostToDevice, plan.stream_small),
               "memcpy warp_shapes");

    // Upload mems
    check_cuda(cudaMalloc(&plan.d_small_mems,
                          ps.total_descs * sizeof(WarpMem)), "malloc warp_mems");
    check_cuda(cudaMemcpyAsync(plan.d_small_mems, h_mems.data(),
                               ps.total_descs * sizeof(WarpMem),
                               cudaMemcpyHostToDevice, plan.stream_small),
               "memcpy warp_mems");

    // Upload warp prefix sum
    check_cuda(cudaMalloc(&plan.d_warp_prefix,
                          (ps.total_descs + 1) * sizeof(int32_t)), "malloc warp_prefix");
    check_cuda(cudaMemcpyAsync(plan.d_warp_prefix, ps.h_warp_prefix.data(),
                               (ps.total_descs + 1) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, plan.stream_small),
               "memcpy warp_prefix");
}

// Per-contracted-dim info for K decomposition.
// Unlike joint K squash, each operand squashes K dims independently —
// merging adjacent K dims when contiguous in that operand.
struct KDimInfo {
    int32_t dim;           // K dimension size
    int64_t stride_a;      // stride in A
    int64_t stride_b;      // stride in B
};

// Build raw (un-squashed) K dim info from ContractionConfig.
inline int build_k_dims(
    const ContractionConfig& config,
    const dim_t* A_dims, dim_t A_rank,
    const dim_t* B_dims, dim_t B_rank,
    KDimInfo* out)
{
    for (int i = 0; i < config.num_contracted; ++i) {
        out[i].dim = A_dims[config.contracted_pos_A[i]];
        out[i].stride_a = compute_dim_stride(A_dims, A_rank, config.contracted_pos_A[i]);
        out[i].stride_b = compute_dim_stride(B_dims, B_rank, config.contracted_pos_B[i]);
    }
    return config.num_contracted;
}

// Build the GridDivmod arrays for one operand's (free, K_tile) load.
struct OperandLoadInfo {
    int rank;
    GridDivmod divmod[VOL_MAX_RANK];
    int64_t stride[VOL_MAX_RANK];
    int32_t tile_volume;
};

// Build operand load info for volume GEMM.
// The flat index ordering is [free_dims..., K_tile_dims...] with K innermost,
// so that idx / K_tile_volume = m and idx % K_tile_volume = k_local.
// k_strides: per-K-dim stride in THIS operand (not shared with other operand).
inline OperandLoadInfo build_operand_load_info(
    const int32_t* free_pos, int num_free,
    const int32_t* k_tile_shapes, const int64_t* k_strides, int k_rank,
    const dim_t* block_dims, dim_t block_rank)
{
    // Fixed logical order (innermost first for divmod decomposition):
    //   K_tile dims (innermost) ... then free dims (outermost)
    // This ensures idx / K_tile_volume = flat_m, idx % K_tile_volume = flat_k.
    struct DimEntry {
        int32_t shape;
        int64_t stride;
    };
    DimEntry entries[VOL_MAX_RANK];
    int n = 0;

    // K tile dims first (innermost) — squash adjacent contiguous dims.
    // Divmod chain needs innermost first (row-major convention), so we reverse.
    {
        auto sq = layout::squash_dimensions<VOL_MAX_RANK>(k_tile_shapes, k_strides, k_rank);
        for (int i = sq.rank - 1; i >= 0; --i) {
            assert(n < VOL_MAX_RANK && "K dims exceed VOL_MAX_RANK");
            entries[n].shape = static_cast<int32_t>(sq.shapes[i]);
            entries[n].stride = sq.strides[i];
            n++;
        }
    }

    // Free dims next (outermost) — squash, then reverse for divmod chain.
    {
        int32_t free_shapes[MAX_RANK];
        int64_t free_strides[MAX_RANK];
        for (int i = 0; i < num_free; ++i) {
            free_shapes[i] = static_cast<int32_t>(block_dims[free_pos[i]]);
            free_strides[i] = compute_dim_stride(block_dims, block_rank, free_pos[i]);
        }
        auto sq = layout::squash_dimensions<VOL_MAX_RANK>(free_shapes, free_strides, num_free);
        for (int i = sq.rank - 1; i >= 0; --i) {
            assert(n < VOL_MAX_RANK && "free dims exceed VOL_MAX_RANK");
            entries[n].shape = static_cast<int32_t>(sq.shapes[i]);
            entries[n].stride = sq.strides[i];
            n++;
        }
    }

    // Degenerate safety: all dims were size 1
    if (n == 0) { entries[0] = {1, 0}; n = 1; }

    // Build divmod chain: divmod[0] is innermost (first K dim)
    OperandLoadInfo out{};
    out.rank = n;
    out.tile_volume = 1;
    for (int i = 0; i < n; ++i) {
        out.divmod[i] = make_grid_divmod(static_cast<uint32_t>(entries[i].shape));
        out.stride[i] = entries[i].stride;
        out.tile_volume *= entries[i].shape;
    }
    for (int i = n; i < VOL_MAX_RANK; ++i) {
        out.divmod[i] = make_grid_divmod(1);
        out.stride[i] = 0;
    }

    return out;
}

// Build VolumeShape geometry (address-independent, no GPU alloc).
// Dedup pair shape keys, compute divmod/strides per unique shape.
// Stores result in plan.prepared_medium for later bind_medium_mems.
template <typename T>
bool prepare_medium_shapes(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    const auto& problems = plan.work.medium.problems;
    const int n_probs = static_cast<int>(problems.size());
    if (n_probs == 0) return true;

    // Dedup pair shape keys.
    std::vector<uint64_t> pair_keys(n_probs);
    for (int i = 0; i < n_probs; ++i)
        pair_keys[i] = make_pair_shape_key(
            plan.shape_map_A.block_to_shape[problems[i].block_idx_A],
            plan.shape_map_B.block_to_shape[problems[i].block_idx_B]);

    std::vector<uint32_t> key_indices(n_probs);
    std::vector<int> first_occ(n_probs);
    int num_unique = pair_key_dedup(pair_keys.data(), n_probs,
                                    key_indices.data(), first_occ.data());

    auto& pm = plan.prepared_medium;
    pm.h_shapes.reserve(num_unique);
    pm.shape_strides.resize(num_unique);

    int max_smem = 0, max_tc = 0;

    for (int u = 0; u < num_unique; ++u) {
        const auto& prob = problems[first_occ[u]];
        const dim_t* A_dims = plan.shape_map_A.shape_dims(
            plan.shape_map_A.block_to_shape[prob.block_idx_A]);
        const dim_t* B_dims = plan.shape_map_B.shape_dims(
            plan.shape_map_B.block_to_shape[prob.block_idx_B]);

        KDimInfo k_dims[MAX_RANK];
        int num_k = build_k_dims(plan.config, A_dims, A_gpu.rank,
                                  B_dims, B_gpu.rank, k_dims);

        int32_t k_shapes[MAX_RANK];
        int64_t k_strides_A[MAX_RANK], k_strides_B[MAX_RANK];
        int K = 1;
        for (int d = 0; d < num_k; ++d) {
            k_shapes[d] = k_dims[d].dim;
            k_strides_A[d] = k_dims[d].stride_a;
            k_strides_B[d] = k_dims[d].stride_b;
            K *= k_dims[d].dim;
        }

        int64_t vol = static_cast<int64_t>(prob.M) * K + static_cast<int64_t>(prob.N) * K;
        if (vol > vol_threshold<T>() || vol_tile_count(prob.M, prob.N) > VOL_MAX_TILES)
            return false;

        OperandLoadInfo A_info = build_operand_load_info(
            plan.config.free_pos_A, plan.config.num_free_A,
            k_shapes, k_strides_A, num_k, A_dims, A_gpu.rank);
        OperandLoadInfo B_info = build_operand_load_info(
            plan.config.free_pos_B, plan.config.num_free_B,
            k_shapes, k_strides_B, num_k, B_dims, B_gpu.rank);

        if (A_info.rank > VOL_MAX_RANK || B_info.rank > VOL_MAX_RANK)
            return false;

        VolumeShape vs{};
        vs.M = static_cast<uint16_t>(prob.M);
        vs.N = static_cast<uint16_t>(prob.N);
        vs.K = static_cast<uint16_t>(K);

        vs.A_rank = static_cast<int8_t>(A_info.rank);
        for (int i = 0; i < VOL_MAX_RANK; ++i) {
            vs.A_divmod[i] = A_info.divmod[i];
            vs.A_stride[i] = static_cast<uint16_t>(A_info.stride[i]);
        }
        vs.A_volume = static_cast<uint16_t>(A_info.tile_volume);

        vs.B_rank = static_cast<int8_t>(B_info.rank);
        for (int i = 0; i < VOL_MAX_RANK; ++i) {
            vs.B_divmod[i] = B_info.divmod[i];
            vs.B_stride[i] = static_cast<uint16_t>(B_info.stride[i]);
        }
        vs.B_volume = static_cast<uint16_t>(B_info.tile_volume);

        vs.K_divmod = make_grid_divmod(static_cast<uint32_t>(K));

        pm.shape_strides[u].batch_stride_a = 1;
        pm.shape_strides[u].batch_stride_b = 1;
        for (int32_t d = plan.config.num_batch; d < A_gpu.rank; ++d)
            pm.shape_strides[u].batch_stride_a *= A_dims[d];
        for (int32_t d = plan.config.num_batch; d < B_gpu.rank; ++d)
            pm.shape_strides[u].batch_stride_b *= B_dims[d];
        pm.shape_strides[u].batch_stride_c = static_cast<int64_t>(prob.M) * prob.N;

        pm.h_shapes.push_back(vs);

        int se = volume_smem_elems(prob.M, prob.N, K);
        int tc = vol_tile_count(prob.M, prob.N);
        if (se > max_smem) max_smem = se;
        if (tc > max_tc) max_tc = tc;
    }

    pm.key_indices = std::move(key_indices);
    pm.num_shapes = static_cast<int>(pm.h_shapes.size());

    int total = 0;
    for (int p = 0; p < n_probs; ++p)
        total += problems[p].batch_size;
    pm.total_descs = total;
    pm.max_smem_elems = max_smem;
    pm.max_tiles = max_tc;

    plan.medium_vol_num_shapes = pm.num_shapes;
    plan.medium_vol_max_smem_elems = pm.max_smem_elems;
    plan.medium_vol_max_tiles = pm.max_tiles;

    // Group medium problems by output block (CSR flat arrays).
    // bind_medium_mems iterates in this order to produce C-pointer-sorted
    // mems without an O(n log n) sort.
    const int64_t nob = plan.accum_map.num_output_blocks;
    std::vector<int32_t> ob_count(static_cast<size_t>(nob), 0);
    for (int p = 0; p < n_probs; ++p)
        ob_count[static_cast<size_t>(
            plan.accum_map.h_pair_output_block_idx[problems[p].pair_idx])]++;

    pm.ob_offsets.resize(static_cast<size_t>(nob) + 1);
    pm.ob_offsets[0] = 0;
    for (int64_t ob = 0; ob < nob; ++ob)
        pm.ob_offsets[static_cast<size_t>(ob) + 1] =
            pm.ob_offsets[static_cast<size_t>(ob)] +
            ob_count[static_cast<size_t>(ob)];

    pm.ob_prob_idx.resize(n_probs);
    pm.ob_batch_sizes.resize(n_probs);
    std::fill(ob_count.begin(), ob_count.end(), 0);
    for (int p = 0; p < n_probs; ++p) {
        const size_t ob = static_cast<size_t>(
            plan.accum_map.h_pair_output_block_idx[problems[p].pair_idx]);
        const int slot = pm.ob_offsets[ob] + ob_count[ob]++;
        pm.ob_prob_idx[slot] = p;
        pm.ob_batch_sizes[slot] = problems[p].batch_size;
    }

    return true;
}

// Fill VolumeMem with absolute GPU pointers and upload shapes+mems to GPU.
// Uses prepared state from prepare_medium_shapes.
template <typename T>
bool bind_medium_mems(
    PipelinePlan<T>& plan,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev)
{
    const auto& pm = plan.prepared_medium;
    const auto& problems = plan.work.medium.problems;
    const int n_probs = static_cast<int>(problems.size());
    if (n_probs == 0) return true;
    if (pm.key_indices.size() != static_cast<size_t>(n_probs)) {
        std::cerr << "bind_medium_mems: pipeline_prepare not called\n";
        return false;
    }

    // Emit mems in output-block order (from prepare's CSR grouping).
    // Within each OB, iterate batch elements then pairs so that mems
    // sharing the same C pointer are consecutive — same result as sorting
    // by C pointer, but O(n) instead of O(n log n).
    std::vector<VolumeMem> h_mems;
    h_mems.reserve(pm.total_descs);

    const int64_t nob = plan.accum_map.num_output_blocks;
    for (int64_t ob = 0; ob < nob; ++ob) {
        const int begin = pm.ob_offsets[static_cast<size_t>(ob)];
        const int end = pm.ob_offsets[static_cast<size_t>(ob) + 1];
        if (begin == end) continue;

        const bool needs_atomic =
            (plan.accum_map.h_ob_tier_flags[ob] & kOBMultiMedium) != 0;

        // Find max batch_size across problems in this OB
        int max_batch = 0;
        for (int i = begin; i < end; ++i) {
            if (pm.ob_batch_sizes[i] > max_batch)
                max_batch = pm.ob_batch_sizes[i];
        }

        // Emit: for each batch element, then each problem in this OB
        for (int b = 0; b < max_batch; ++b) {
            const size_t group_start = h_mems.size();
            for (int i = begin; i < end; ++i) {
                if (b >= pm.ob_batch_sizes[i]) continue;
                const int p = pm.ob_prob_idx[i];
                const auto& prob = problems[p];
                const auto& ss = pm.shape_strides[pm.key_indices[p]];

                VolumeMem vm;
                vm.A = A_dev.d_values + prob.offset_A + b * ss.batch_stride_a;
                vm.B = B_dev.d_values + prob.offset_B + b * ss.batch_stride_b;
                vm.C = plan.d_output +
                       plan.accum_map.h_dest_offsets[prob.pair_idx] +
                       b * ss.batch_stride_c;
                vm.shape_idx = pm.key_indices[p];
                vm.use_atomic = needs_atomic ? 1 : 0;
                vm.flush_output = 0;
                h_mems.push_back(vm);
            }
            // Last mem in each same-C group gets flush
            if (!h_mems.empty() && h_mems.size() > group_start)
                h_mems.back().flush_output = 1;
        }
    }

    plan.medium_vol_num_problems = static_cast<int>(h_mems.size());

    if (plan.medium_vol_num_problems > 0) {
        // Upload shapes
        check_cuda(cudaMalloc(&plan.d_medium_vol_shapes,
                              pm.h_shapes.size() * sizeof(VolumeShape)),
                   "malloc d_medium_vol_shapes");
        check_cuda(cudaMemcpyAsync(plan.d_medium_vol_shapes, pm.h_shapes.data(),
                                    pm.h_shapes.size() * sizeof(VolumeShape),
                                    cudaMemcpyHostToDevice, plan.stream_medium),
                   "memcpy d_medium_vol_shapes");

        // Upload mems
        check_cuda(cudaMalloc(&plan.d_medium_vol_mems,
                              h_mems.size() * sizeof(VolumeMem)),
                   "malloc d_medium_vol_mems");
        check_cuda(cudaMemcpyAsync(plan.d_medium_vol_mems, h_mems.data(),
                                    h_mems.size() * sizeof(VolumeMem),
                                    cudaMemcpyHostToDevice, plan.stream_medium),
                   "memcpy d_medium_vol_mems");
    }
    return true;
}

// ============================================================================
// Phase 1: CPU planning + small GPU allocs. No big buffer allocation.
// After this returns, call pipeline_query_workspace() to get buffer sizes,
// then pipeline_allocate() to bind buffers and finalize GPU setup.
// ============================================================================
template <typename T>
bool pipeline_plan(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const EinsumIndexSets& index_sets,
    const std::string& output_labels = "",
    const DispatchConfig& dispatch_config = DispatchConfig(),
    PipelineTimingBreakdown* timing = nullptr,
    const OutputBlockMap* output_blocks = nullptr)
{
    using clock = std::chrono::high_resolution_clock;

    plan.timing = timing;
    plan.output_beta = dispatch_config.output_beta;

    plan.config = build_contraction_config(
        index_sets, A_gpu.labels, B_gpu.labels, output_labels);
    plan.output_rank = plan.config.output_rank;

    // Enforce batch-leftmost constraint: batch dimensions must occupy positions
    // 0..num_batch-1 in both operands. This guarantees that volume and warp
    // kernels can handle interleaved layouts natively without pre-transpose.
    if (plan.config.num_batch > 0) {
        if (!batch_dims_are_leading(plan.config.batch_pos_A, plan.config.num_batch)) {
            std::cerr << "pipeline_plan: batch dims not leading in A\n";
            return false;
        }
        if (!batch_dims_are_leading(plan.config.batch_pos_B, plan.config.num_batch)) {
            std::cerr << "pipeline_plan: batch dims not leading in B\n";
            return false;
        }
    }

    // Host-side hash join: find matching block pairs (no GPU alloc)
    auto t_stage = clock::now();
    plan.pairs_host = find_matching_pairs(A_gpu, B_gpu, plan.config);
    plan.num_pairs = static_cast<int64_t>(plan.pairs_host.size());

    if (plan.num_pairs == 0) {
        if (timing) {
            double pp = detail_timing::ms_since(t_stage);
            *timing = PipelineTimingBreakdown{};
            timing->preprocess_ms = pp;
        }
        return true;
    }

    // Compute output offsets via prefix sum
    {
        int64_t offset = 0;
        for (auto& p : plan.pairs_host) {
            p.output_offset = offset;
            offset += p.output_volume;
        }
        plan.total_pair_output_size = offset;
    }

    if (timing) { timing->preprocess_ms = detail_timing::ms_since(t_stage); }

    // Layout planning (CPU only) — must precede partition_by_threshold for gemm_mode.
    plan.layout_plan = plan_gemm_layout(plan.config, A_gpu.rank, B_gpu.rank);

    plan.gemm_mode = plan.layout_plan.mode_after_transpose;
    const bool physical_transpose_needed = plan.layout_plan.plan_A.transpose ||
                                          plan.layout_plan.plan_B.transpose;

    // Single classification pass — no device pointers needed, pure tier assignment.
    // Volume eligibility is always true; per-problem M,N and volume-threshold
    // checks in partition_by_threshold do the real gating, and the post-squash
    // rank guard in setup_medium_volume_descs catches any remaining cases.
    plan.work = partition_by_threshold(
        A_gpu, B_gpu, plan.pairs_host, plan.config,
        plan.gemm_mode, /*volume_eligible=*/true,
        /*needs_transpose=*/physical_transpose_needed,
        dispatch_config.vol_max_tiles);

    // Build work_tier: 0 = warp, 1 = volume, 2 = CUTLASS (for AccumulationMap flags).
    std::vector<uint8_t> work_tier(static_cast<size_t>(plan.num_pairs), 2);
    for (const auto& prob : plan.work.small.problems)
        work_tier[static_cast<size_t>(prob.pair_idx)] = 0;
    for (const auto& prob : plan.work.medium.problems)
        work_tier[static_cast<size_t>(prob.pair_idx)] = 1;

    t_stage = clock::now();
    if (output_blocks) {
        // Fast O(n) path: output block structure pre-computed
        plan.accum_map = compute_accumulation_map(
            plan.pairs_host.data(), plan.num_pairs, A_gpu, B_gpu, plan.config,
            *output_blocks, work_tier.data());
    } else {
        // Discovery path: O(n log n) sort to find output blocks
        plan.accum_map = compute_accumulation_map(
            plan.pairs_host.data(), plan.num_pairs, A_gpu, B_gpu, plan.config,
            work_tier.data());
    }
    if (timing) { timing->accum_map_ms = detail_timing::ms_since(t_stage); }

    // Per-tensor block shape dedup: blocks with identical dims share a shape ID.
    // Downstream desc setup uses this to squash/compute geometry once per unique
    // shape rather than once per block.
    plan.shape_map_A = build_block_shape_map(
        A_gpu.block_dims.data(), A_gpu.num_blocks, A_gpu.rank);
    plan.shape_map_B = build_block_shape_map(
        B_gpu.block_dims.data(), B_gpu.num_blocks, B_gpu.rank);

    // Build pair_tier: 0 = small/volume (writes directly to d_output with atomics),
    //                  2 = large/CUTLASS (writes to temp buffer, eligible for direct-write)
    // Volume tier (medium) writes directly to d_output like warp, so pair_tier = 0.
    std::vector<uint8_t> pair_tier(static_cast<size_t>(plan.num_pairs), 0);
    for (const auto& prob : plan.work.large.problems)
        pair_tier[static_cast<size_t>(prob.pair_idx)] = 2;

    // Early streaming partition: needed before temp buffer sizing so that
    // separated pairs can be routed to d_output (skipping temp buffer).
    bool early_can_stream = dispatch_config.transpose_buf_cap > 0
                            && !plan.work.large.empty();

    std::vector<size_t> separated_pair_indices;
    if (early_can_stream) {
        plan.staging_classified = classify_cutlass_work(plan.work);
        merge_small_queues(plan.staging_classified);
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            auto& qwork = plan.staging_classified.queues[qi];
            if (qwork.empty()) continue;
            auto qpart = partition_large_smart(
                qwork, A_gpu, B_gpu, plan.accum_map,
                dispatch_config.transpose_buf_cap);
            compute_pair_separation(qpart, qwork, plan.accum_map);
            if (!qpart.pair_separated.empty()) {
                for (size_t li = 0; li < qwork.problems.size(); ++li) {
                    if (qpart.pair_separated[li]) {
                        size_t pi = static_cast<size_t>(qwork.problems[li].pair_idx);
                        pair_tier[pi] = 0;
                        separated_pair_indices.push_back(pi);
                    }
                }
            }
            plan.streaming.queue_partitions[qi] = std::move(qpart);
        }
    }

    // Right-size temp buffer: only non-direct-write CUTLASS pairs need space.
    const size_t np = static_cast<size_t>(plan.num_pairs);
    const size_t nob = static_cast<size_t>(plan.accum_map.num_output_blocks);

    // Mixed output blocks (both CUTLASS and volume pairs) are detected via
    // h_ob_tier_flags (kOBHasLarge & kOBHasMedium), computed in AccumulationMap.

    std::vector<int64_t> tb_count(nob, 0);
    for (size_t p = 0; p < np; ++p) {
        if (pair_tier[p] != 0)
            tb_count[static_cast<size_t>(
                plan.accum_map.h_pair_output_block_idx[p])]++;
    }

    std::vector<uint8_t> is_direct(np, 0);
    for (int64_t ob = 0; ob < plan.accum_map.num_output_blocks; ++ob) {
        if (tb_count[static_cast<size_t>(ob)] != 1) continue;
        // Mixed blocks: CUTLASS must go through temp buffer, not direct-write
        if ((plan.accum_map.h_ob_tier_flags[static_cast<size_t>(ob)] &
             (kOBHasLarge | kOBHasMedium)) == (kOBHasLarge | kOBHasMedium))
            continue;
        const int64_t begin = plan.accum_map.h_output_pair_offsets[ob];
        const int64_t end = plan.accum_map.h_output_pair_offsets[ob + 1];
        for (int64_t i = begin; i < end; ++i) {
            const int64_t pi = plan.accum_map.h_output_pair_indices[i];
            if (pair_tier[static_cast<size_t>(pi)] != 0) {
                is_direct[static_cast<size_t>(pi)] = 1;
                break;
            }
        }
    }

    int64_t compact_offset = 0;
    if (early_can_stream) {
        // Streaming: reuse temp slots across batches for the same output block.
        // Pairs in different (queue, batch) groups are serialized, so they can
        // share a temp slot and accumulate via CUTLASS beta=1.

        // Build pair_idx → (queue * 10000 + batch) concurrent-group key
        std::unordered_map<size_t, int> pair_qb;
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            const auto& qwork = plan.staging_classified.queues[qi];
            const auto& qpart = plan.streaming.queue_partitions[qi];
            for (int bi = 0; bi < static_cast<int>(qpart.batches.size()); ++bi) {
                int key = qi * 10000 + bi;
                for (int li : qpart.batches[bi].problem_indices) {
                    pair_qb[static_cast<size_t>(qwork.problems[li].pair_idx)] = key;
                }
            }
        }

        // Per output block: find max concurrent pairs, assign shared offsets
        for (int64_t ob = 0; ob < plan.accum_map.num_output_blocks; ++ob) {
            const int64_t begin = plan.accum_map.h_output_pair_offsets[ob];
            const int64_t end = plan.accum_map.h_output_pair_offsets[ob + 1];

            // Collect non-direct CUTLASS pairs for this output block
            std::vector<size_t> ob_pairs;
            for (int64_t i = begin; i < end; ++i) {
                size_t pi = static_cast<size_t>(plan.accum_map.h_output_pair_indices[i]);
                if (pair_tier[pi] != 0 && !is_direct[pi])
                    ob_pairs.push_back(pi);
            }
            if (ob_pairs.empty()) continue;

            int64_t volume = plan.pairs_host[ob_pairs[0]].output_volume;

            // Count pairs per concurrent group; find max
            std::unordered_map<int, int> group_count;
            for (size_t pi : ob_pairs) {
                auto it = pair_qb.find(pi);
                int key = (it != pair_qb.end()) ? it->second : -1;
                group_count[key]++;
            }
            int max_slots = 0;
            for (const auto& [k, cnt] : group_count)
                max_slots = std::max(max_slots, cnt);

            // Assign offsets: pairs in same (queue, batch) get sequential slots
            int64_t block_base = compact_offset;
            compact_offset += static_cast<int64_t>(max_slots) * volume;

            std::unordered_map<int, int> group_next_slot;
            for (size_t pi : ob_pairs) {
                auto it = pair_qb.find(pi);
                int key = (it != pair_qb.end()) ? it->second : -1;
                int slot = group_next_slot[key]++;
                plan.pairs_host[pi].output_offset =
                    block_base + static_cast<int64_t>(slot) * volume;
            }
        }
    } else {
        // Non-streaming: each pair gets a unique temp slot
        for (size_t p = 0; p < np; ++p) {
            if (pair_tier[p] != 0 && !is_direct[p]) {
                plan.pairs_host[p].output_offset = compact_offset;
                compact_offset += plan.pairs_host[p].output_volume;
            }
        }
    }
    plan.total_pair_output_size = compact_offset;

    // Reduction plan (host-side only; GPU upload deferred to pipeline_allocate)
    plan.reduce_plan = build_reduction_plan(plan.accum_map, pair_tier,
                                            plan.pairs_host.data(),
                                            plan.accum_map.h_ob_tier_flags);
    for (size_t pi : separated_pair_indices) {
        plan.reduce_plan.h_direct_write[pi] = 1;
    }

    // Only CUTLASS needs pre-transpose; volume and warp handle interleaved natively.
    plan.needs_transpose = physical_transpose_needed && !plan.work.large.empty();

    // Compute streaming workspace sizes (CPU only — no GPU allocs)
    plan.streaming.enabled = early_can_stream;
    if (plan.streaming.enabled) {
        int64_t max_buf_cap = 0;
        int64_t max_expanded = 0;
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            if (plan.staging_classified.queues[qi].empty()) continue;
            const auto& qp = plan.streaming.queue_partitions[qi];
            max_buf_cap = std::max(max_buf_cap, qp.max_batch_transpose_volume);
            max_expanded = std::max(max_expanded, qp.max_batch_expanded_problems);
        }
        plan.streaming.buf_cap = max_buf_cap;
        plan.streaming.max_expanded_problems = max_expanded;
    }

    return true;
}

// ============================================================================
// Query workspace requirements after pipeline_plan.
// ============================================================================
template <typename T>
PipelineWorkspaceInfo pipeline_query_workspace(
    const PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    PipelineWorkspaceInfo info;
    constexpr int64_t elem = static_cast<int64_t>(sizeof(T));
    constexpr int64_t ALIGN = 256;
    auto align_up = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };

    info.output_elements = plan.accum_map.total_output_elements;
    info.output_bytes = info.output_elements * elem;

    // Temp buffer (compact: only non-direct-write CUTLASS pairs)
    int64_t ws = 0;
    if (plan.total_pair_output_size > 0)
        ws += align_up(plan.total_pair_output_size * elem, ALIGN);

    // Transpose buffers
    if (plan.streaming.enabled) {
        // Streaming transpose buffers (only when physical transpose needed)
        if (plan.needs_transpose && plan.streaming.buf_cap > 0) {
            int total_batches = 0;
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi)
                total_batches += static_cast<int>(plan.streaming.queue_partitions[qi].batches.size());
            int num_bufs = (total_batches <= 1) ? 1 : 2;
            ws += num_bufs * align_up(plan.streaming.buf_cap * elem, ALIGN);
        }
    } else if (plan.needs_transpose) {
        // Legacy: full-size workspace per transposed operand
        int64_t A_elements = static_cast<int64_t>(A_gpu.get_values_count());
        int64_t B_elements = static_cast<int64_t>(B_gpu.get_values_count());
        if (plan.layout_plan.plan_A.transpose)
            ws += align_up(A_elements * elem, ALIGN);
        if (plan.layout_plan.plan_B.transpose)
            ws += align_up(B_elements * elem, ALIGN);
    }

    info.workspace_bytes = ws;
    return info;
}

// Build per-batch LargeBlockWork for one streaming batch.
// Shared by precompute (pipeline_allocate) and runtime fallback (stage_large_gemm_streamed).
template <typename T>
LargeBlockWork<T> build_streaming_batch_work(
    const PipelinePlan<T>& plan,
    const LargeBlockWork<T>& queue_work,
    const TransposeBatch& batch,
    const std::vector<int64_t>& expanded_start,
    const TransposeFlatMeta& meta_A,
    const TransposeFlatMeta& meta_B,
    T* d_buf,
    T* d_A_values,
    T* d_B_values)
{
    LargeBlockWork<T> bw;
    bw.gemm_mode = queue_work.gemm_mode;

    const auto& pack_offset_A = batch.pack_offset_A;
    const auto& pack_offset_B = batch.pack_offset_B;
    const auto& local_A = batch.local_A;
    const auto& local_B = batch.local_B;

    for (size_t pi_idx = 0; pi_idx < batch.problem_indices.size(); ++pi_idx) {
        int pi = batch.problem_indices[pi_idx];
        const auto& prob = queue_work.problems[pi];
        bw.problems.push_back(prob);

        bool A_use_buf = plan.layout_plan.plan_A.transpose
            && meta_A.kind != TransposeDispatchKind::None
            && !plan.trivial_shapes_A[plan.shape_map_A.block_to_shape[prob.block_idx_A]];
        bool B_use_buf = plan.layout_plan.plan_B.transpose
            && meta_B.kind != TransposeDispatchKind::None
            && !plan.trivial_shapes_B[plan.shape_map_B.block_to_shape[prob.block_idx_B]];
        T* A_base = A_use_buf ? d_buf : d_A_values;
        T* B_base = B_use_buf ? d_buf : d_B_values;
        int64_t A_off = A_use_buf ? pack_offset_A[local_A[pi_idx]] : prob.offset_A;
        int64_t B_off = B_use_buf ? pack_offset_B[local_B[pi_idx]] : prob.offset_B;

        int64_t A_batch_stride = static_cast<int64_t>(prob.M) * prob.K;
        int64_t B_batch_stride = static_cast<int64_t>(prob.K) * prob.N;
        int64_t es = expanded_start[pi];

        for (int64_t b = 0; b < prob.batch_size; ++b) {
            bw.ptr_A.push_back(A_base + A_off + b * A_batch_stride);
            bw.ptr_B.push_back(B_base + B_off + b * B_batch_stride);
            bw.ptr_C.push_back(queue_work.ptr_C[es + b]);
            bw.lda.push_back(prob.lda);
            bw.ldb.push_back(prob.ldb);
            bw.ldc.push_back(prob.ldc);
            bw.ldd.push_back(prob.ldc);
        }
    }
    bw.num_gemm_problems = static_cast<int64_t>(bw.ptr_A.size());
    return bw;
}

// ============================================================================
// Phase 2: Address-independent shape/geometry build. No GPU pointers needed.
// Builds shapes, dedup, tile selection, hypertile index — everything that
// can be reused across GPU buffer reallocations.
// ============================================================================
template <typename T>
void pipeline_prepare(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    if (plan.num_pairs == 0) return;

    // Small tier: shape geometry + warp prefix
    if (!plan.work.small.empty())
        prepare_small_shapes(plan, A_gpu, B_gpu);

    // Medium tier: VolumeShape + divmod
    if (!plan.work.medium.empty()) {
        if (!prepare_medium_shapes(plan, A_gpu, B_gpu)) {
            std::cerr << "pipeline_prepare: prepare_medium_shapes failed "
                         "(partitioner should guarantee volume eligibility)\n";
        }
    }

    // Large tier: classify shapes + select tiles
    if (!plan.work.large.empty()) {
        if (!plan.streaming.enabled) {
            plan.nonstream_classified = classify_cutlass_work(plan.work);
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
                if (plan.nonstream_classified.queues[qi].empty()) continue;
                plan.nonstream_tiles[qi] = best_tile_for_queue(
                    plan.nonstream_classified.queues[qi]);
            }
        } else {
            // Streaming: compute tiles from staging_classified (built in pipeline_plan)
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
                if (plan.staging_classified.queues[qi].empty()) continue;
                plan.streaming.queue_tiles[qi] = best_tile_for_queue(
                    plan.staging_classified.queues[qi]);
            }
        }
    }

    // Transpose: build hypertile index + trivial shapes (address-independent)
    if (plan.needs_transpose) {
        if (plan.layout_plan.plan_A.transpose &&
            plan.layout_plan.plan_A.dispatch_kind == TransposeDispatchKind::GenericPermute) {
            plan.hypertile_index_A = build_hypertile_index<T>(
                A_gpu, plan.layout_plan.plan_A, plan.shape_map_A);
        }
        if (plan.layout_plan.plan_B.transpose &&
            plan.layout_plan.plan_B.dispatch_kind == TransposeDispatchKind::GenericPermute) {
            plan.hypertile_index_B = build_hypertile_index<T>(
                B_gpu, plan.layout_plan.plan_B, plan.shape_map_B);
        }
        if (plan.layout_plan.plan_A.transpose) {
            plan.trivial_shapes_A = compute_trivial_shapes(
                plan.shape_map_A,
                plan.hypertile_index_A.valid ? &plan.hypertile_index_A : nullptr);
        }
        if (plan.layout_plan.plan_B.transpose) {
            plan.trivial_shapes_B = compute_trivial_shapes(
                plan.shape_map_B,
                plan.hypertile_index_B.valid ? &plan.hypertile_index_B : nullptr);
        }
    }
}

// ============================================================================
// Phase 3: Bind workspace buffers, patch GPU pointers, upload to device.
// If d_output/workspace are null, self-allocates (backward-compat behavior).
// ============================================================================
template <typename T>
bool pipeline_allocate(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    T* d_output = nullptr,
    void* workspace = nullptr,
    int64_t workspace_bytes = 0)
{
    if (plan.num_pairs == 0) return true;

    // Snapshot free memory before allocations for accurate peak measurement
    size_t free_before = 0, total_ignore = 0;
    cudaMemGetInfo(&free_before, &total_ignore);

    using alloc_clock = std::chrono::high_resolution_clock;
    double bind_accum = 0;

    // --- GPU uploads deferred from pipeline_plan ---
    double bind_upload = 0, bind_patch = 0, bind_stream = 0;
    auto tb0 = alloc_clock::now();
    upload_accumulation_map(plan.accum_map);
    upload_reduction_plan(plan.reduce_plan);
    bind_upload = detail_timing::ms_since(tb0);

    check_cuda(cudaStreamCreate(&plan.stream_large), "create stream_large");
    check_cuda(cudaStreamCreate(&plan.stream_small), "create stream_small");
    check_cuda(cudaStreamCreate(&plan.stream_medium), "create stream_medium");
    check_cuda(cudaEventCreateWithFlags(&plan.transpose_done_flat, cudaEventDisableTiming),
               "create transpose_done_flat");
    check_cuda(cudaEventCreateWithFlags(&plan.transpose_done_tiled, cudaEventDisableTiming),
               "create transpose_done_tiled");

    constexpr int64_t elem = static_cast<int64_t>(sizeof(T));
    constexpr int64_t ALIGN = 256;

    // --- Bind or allocate d_output ---
    if (d_output) {
        plan.d_output = d_output;
        plan.owns_output = false;
    } else {
        check_cuda(cudaMalloc(&plan.d_output, plan.accum_map.total_output_elements * elem),
                   "malloc final_output");
        plan.owns_output = true;
    }

    if (plan.output_beta == 0) {
        check_cuda(cudaMemset(plan.d_output, 0,
                   plan.accum_map.total_output_elements * elem), "memset final_output");
    }

    // --- Sub-allocate workspace or self-allocate ---
    auto bump = [&](int64_t bytes) -> void* {
        if (!workspace) return nullptr;
        uintptr_t p = reinterpret_cast<uintptr_t>(workspace);
        p = (p + ALIGN - 1) & ~static_cast<uintptr_t>(ALIGN - 1);
        void* result = reinterpret_cast<void*>(p);
        int64_t used = static_cast<int64_t>(p - reinterpret_cast<uintptr_t>(workspace)) + bytes;
        workspace = reinterpret_cast<void*>(p + bytes);
        workspace_bytes -= used;
        return result;
    };

    plan.owns_workspace = (workspace == nullptr);

    // Temp buffer
    if (plan.total_pair_output_size > 0) {
        int64_t temp_bytes = plan.total_pair_output_size * elem;
        if (workspace) {
            plan.d_temp_buffer = static_cast<T*>(bump(temp_bytes));
        } else {
            check_cuda(cudaMalloc(&plan.d_temp_buffer, temp_bytes), "malloc temp_buffer");
        }
        check_cuda(cudaMemset(plan.d_temp_buffer, 0, temp_bytes), "memset temp_buffer");
    }

    // Init transposed values
    plan.tv.d_A = A_dev.d_values;
    plan.tv.d_B = B_dev.d_values;
    plan.tv.d_workspace_A = nullptr;
    plan.tv.d_workspace_B = nullptr;
    plan.tv.gemm_mode = GemmMode::NN;

    // --- Patch GPU pointers into large tier (needs d_output + d_temp_buffer) ---
    auto tb1 = alloc_clock::now();
    if (!plan.work.large.empty()) {
        build_large_pointers(plan.work.large, plan.pairs_host,
            A_dev.d_values, B_dev.d_values, plan.d_temp_buffer,
            plan.d_output, plan.accum_map.h_dest_offsets,
            plan.reduce_plan.h_direct_write);

        if (!plan.streaming.enabled) {
            // Distribute pointers into pre-classified queues, build CUTLASS runners
            distribute_large_pointers(plan.nonstream_classified, plan.work.large);
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
                auto& qwork = plan.nonstream_classified.queues[qi];
                if (qwork.empty()) continue;
                plan.nonstream_runners[qi] = prepare_cutlass_gemm_dispatch(
                    qwork, T(1), T(1), plan.nonstream_tiles[qi], plan.stream_large);
            }
        }
    }

    // --- Bind small/medium mems (patch GPU pointers + upload) ---
    if (!plan.work.small.empty())
        bind_small_mems(plan, A_dev, B_dev);

    if (!plan.work.medium.empty()) {
        if (!bind_medium_mems(plan, A_dev, B_dev)) {
            std::cerr << "pipeline_allocate: bind_medium_mems failed\n";
            return false;
        }
    }
    bind_patch = detail_timing::ms_since(tb1);

    // --- Streaming / transpose workspace setup ---
    auto tb2 = alloc_clock::now();
    if (plan.streaming.enabled) {
        // Distribute pointer arrays, move staging → streaming queue_work
        distribute_large_pointers(plan.staging_classified, plan.work.large);
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            plan.streaming.queue_work[qi] = std::move(plan.staging_classified.queues[qi]);
            // tiles already set in pipeline_prepare
        }
        plan.staging_classified = ShapeClassifiedWork<T>{};

        // Allocate or bind streaming transpose buffers (only when transpose needed)
        if (plan.needs_transpose && plan.streaming.buf_cap > 0) {
            int64_t buf_bytes = plan.streaming.buf_cap * elem;
            int total_batches = 0;
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi)
                total_batches += static_cast<int>(plan.streaming.queue_partitions[qi].batches.size());
            int num_bufs = (total_batches <= 1) ? 1 : 2;
            for (int b = 0; b < num_bufs; ++b) {
                if (workspace) {
                    plan.streaming.d_buf[b] = static_cast<T*>(bump(buf_bytes));
                } else {
                    check_cuda(cudaMalloc(&plan.streaming.d_buf[b], buf_bytes),
                               "malloc streaming transpose buffer");
                }
            }
        }

        // Streaming streams + events
        check_cuda(cudaStreamCreate(&plan.streaming.streams[0]), "create streaming stream 0");
        check_cuda(cudaStreamCreate(&plan.streaming.streams[1]), "create streaming stream 1");
        check_cuda(cudaEventCreateWithFlags(&plan.streaming.gemm_done[0], cudaEventDisableTiming),
                   "create streaming event 0");
        check_cuda(cudaEventCreateWithFlags(&plan.streaming.gemm_done[1], cudaEventDisableTiming),
                   "create streaming event 1");
        check_cuda(cudaEventCreateWithFlags(&plan.streaming.copy_done[0], cudaEventDisableTiming),
                   "create streaming copy_done event 0");
        check_cuda(cudaEventCreateWithFlags(&plan.streaming.copy_done[1], cudaEventDisableTiming),
                   "create streaming copy_done event 1");

        // Pre-allocate CUTLASS arg buffers (shared by precomputed + runtime paths)
        if (plan.streaming.max_expanded_problems > 0) {
            plan.streaming.args[0] = allocate_cutlass_args(
                static_cast<int>(plan.streaming.max_expanded_problems));
            plan.streaming.args[1] = allocate_cutlass_args(
                static_cast<int>(plan.streaming.max_expanded_problems));
        }

        // Pre-allocate transpose descriptor buffers (only when transpose needed)
        if (plan.needs_transpose) {
            for (int slot = 0; slot < 2; ++slot) {
                if (plan.layout_plan.plan_A.transpose)
                    plan.streaming.precomp_bufs[slot][0].allocate(A_gpu.num_blocks);
                if (plan.layout_plan.plan_B.transpose)
                    plan.streaming.precomp_bufs[slot][1].allocate(B_gpu.num_blocks);
            }
        }

        // Precompute first ≤2 batches: fill CUTLASS args + transpose metadata
        // during setup so execute is pure kernel launches.
        // Cases with >2 batches fall through to runtime double-buffer path.
        {
            int total_streaming_batches = 0;
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi)
                for (const auto& b : plan.streaming.queue_partitions[qi].batches)
                    if (!b.problem_indices.empty()) ++total_streaming_batches;

            if (total_streaming_batches <= 2) {
                int gslot = 0;
                for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
                    const auto& partition = plan.streaming.queue_partitions[qi];
                    const auto& qlarge = plan.streaming.queue_work[qi];
                    TileShape tile = plan.streaming.queue_tiles[qi];
                    if (partition.batches.empty() || qlarge.empty()) continue;

                    std::vector<int64_t> expanded_start(qlarge.problems.size());
                    int64_t eidx = 0;
                    for (size_t j = 0; j < qlarge.problems.size(); ++j) {
                        expanded_start[j] = eidx;
                        eidx += qlarge.problems[j].batch_size;
                    }

                    for (size_t bi = 0; bi < partition.batches.size(); ++bi) {
                        const auto& batch = partition.batches[bi];
                        if (batch.problem_indices.empty()) continue;

                        int slot = gslot;
                        T* d_buf = plan.streaming.d_buf[slot];

                        TransposeFlatMeta meta_A, meta_B;
                        if (plan.layout_plan.plan_A.transpose) {
                            meta_A = assemble_hypertile_batch<T>(
                                batch.unique_blocks_A, A_gpu, plan.shape_map_A,
                                plan.hypertile_index_A, plan.trivial_shapes_A,
                                batch.pack_offset_A,
                                &plan.streaming.precomp_bufs[slot][0], 0);
                        }
                        if (plan.layout_plan.plan_B.transpose) {
                            meta_B = assemble_hypertile_batch<T>(
                                batch.unique_blocks_B, B_gpu, plan.shape_map_B,
                                plan.hypertile_index_B, plan.trivial_shapes_B,
                                batch.pack_offset_B,
                                &plan.streaming.precomp_bufs[slot][1], 0);
                        }
                        plan.streaming.precomp_meta[slot][0] = meta_A;
                        plan.streaming.precomp_meta[slot][1] = meta_B;

                        auto batch_work = build_streaming_batch_work(
                            plan, qlarge, batch, expanded_start,
                            meta_A, meta_B, d_buf,
                            A_dev.d_values, B_dev.d_values);

                        fill_cutlass_args_dispatch(
                            plan.streaming.args[slot], batch_work,
                            T(1), T(1), tile, 0);

                        ++gslot;
                    }
                }
                plan.streaming.num_precomp = gslot;
                plan.streaming.precomputed = true;
            }
        }

    } else if (plan.needs_transpose) {
        // Non-streaming transpose: assemble transpose batch + workspace alloc
        // (hypertile_index + trivial_shapes already built in pipeline_prepare)
        auto build_transpose = [&](const GPUSparseInput<T>& gpu, const auto& op_plan,
                                   const BlockShapeMap& smap,
                                   const HypertileIndex& idx,
                                   const std::vector<uint8_t>& trivial_shapes,
                                   TransposeFlatMeta& precomp, T*& d_workspace,
                                   int64_t total_elements) {
            if (!op_plan.transpose) return;
            std::vector<int> all_blocks(gpu.num_blocks);
            std::iota(all_blocks.begin(), all_blocks.end(), 0);
            precomp = assemble_hypertile_batch<T>(
                all_blocks, gpu, smap, idx, trivial_shapes,
                {}, nullptr, plan.stream_large);
            if (precomp.kind != TransposeDispatchKind::None) {
                int64_t ws_bytes = total_elements * elem;
                if (workspace) {
                    d_workspace = static_cast<T*>(bump(ws_bytes));
                } else {
                    check_cuda(cudaMalloc(&d_workspace, ws_bytes),
                               "malloc transpose workspace");
                }
            }
        };
        build_transpose(A_gpu, plan.layout_plan.plan_A, plan.shape_map_A,
                      plan.hypertile_index_A, plan.trivial_shapes_A,
                      plan.transpose_precomp_A,
                      plan.tv.d_workspace_A, A_dev.total_elements);
        build_transpose(B_gpu, plan.layout_plan.plan_B, plan.shape_map_B,
                      plan.hypertile_index_B, plan.trivial_shapes_B,
                      plan.transpose_precomp_B,
                      plan.tv.d_workspace_B, B_dev.total_elements);
    }

    bind_stream = detail_timing::ms_since(tb2);

    // Measure actual pipeline allocation via cudaMemGetInfo delta
    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total_ignore);
    plan.measured_alloc_bytes = static_cast<int64_t>(free_before - free_after);

    if (plan.timing) {
        populate_timing_stats(*plan.timing, plan, A_dev, B_dev, A_gpu, B_gpu);
        plan.timing->bind_upload_ms = bind_upload;
        plan.timing->bind_patch_ms = bind_patch;
        plan.timing->bind_stream_ms = bind_stream;
        plan.timing->bind_ms = bind_upload + bind_patch + bind_stream;
    }

    cudaDeviceSynchronize();
    return true;
}

// ============================================================================
// Backward-compatible one-call setup (self-allocates all buffers).
// ============================================================================
template <typename T>
bool pipeline_setup(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    const EinsumIndexSets& index_sets,
    const std::string& output_labels = "",
    const DispatchConfig& dispatch_config = DispatchConfig(),
    PipelineTimingBreakdown* timing = nullptr,
    const OutputBlockMap* output_blocks = nullptr)
{
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    if (!pipeline_plan(plan, A_gpu, B_gpu,
                       index_sets, output_labels, dispatch_config, timing,
                       output_blocks))
        return false;
    if (timing) timing->plan_ms = detail_timing::ms_since(t0);
    auto t1 = clock::now();
    pipeline_prepare(plan, A_gpu, B_gpu);
    if (timing) timing->prepare_ms = detail_timing::ms_since(t1);
    auto t2 = clock::now();
    bool ok = pipeline_allocate(plan, A_gpu, B_gpu, A_dev, B_dev);
    if (timing) timing->allocate_ms = detail_timing::ms_since(t2);
    return ok;
}

#include "pipeline_stages.hpp"

template <typename T>
void pipeline_execute(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev)
{
    using clock = std::chrono::high_resolution_clock;
    auto* timing = plan.timing;

    if (plan.num_pairs == 0) return;

    auto t_total = clock::now();
    auto t_stage = t_total;

    // --- Transpose + pointer patching ---
    if (plan.streaming.enabled) {
        // Streaming mode: transpose (if any) handled per-batch inside
        // stage_large_gemm_streamed_perqueue; nothing to do here.
        if (timing) { timing->transpose_ms = 0.0; }
    } else if (plan.needs_transpose) {
        t_stage = clock::now();
        stage_transpose(plan, A_dev, B_dev);

        // Cross-stream ordering: GEMM streams wait for transpose completion
        cudaStreamWaitEvent(plan.stream_large, plan.transpose_done_flat);
        cudaStreamWaitEvent(plan.stream_medium, plan.transpose_done_flat);
        cudaStreamWaitEvent(plan.stream_medium, plan.transpose_done_tiled);
        cudaStreamWaitEvent(plan.stream_small, plan.transpose_done_flat);
        cudaStreamWaitEvent(plan.stream_small, plan.transpose_done_tiled);

        if (timing) {
            cudaEventSynchronize(plan.transpose_done_flat);
            cudaEventSynchronize(plan.transpose_done_tiled);
            timing->transpose_ms = detail_timing::ms_since(t_stage);
        }

        stage_patch_pointers(plan, A_dev, B_dev);
    } else {
        if (timing) { timing->transpose_ms = 0.0; }
    }

    // --- GEMM execution (3 tiers, parallel streams) ---
    t_stage = clock::now();

    if (plan.streaming.enabled) {
        // Launch warp + volume first on separate streams.
        // Streaming CUTLASS may block host (runtime fallback has inter-batch syncs),
        // so enqueue independent work before it for GPU overlap.
        stage_medium_gemm(plan, plan.stream_medium);
        stage_small_gemm(plan, plan.stream_small);
        stage_large_gemm_streamed_perqueue(plan, A_gpu, B_gpu, A_dev, B_dev);
    } else {
        stage_large_gemm(plan, plan.stream_large);
        stage_medium_gemm(plan, plan.stream_medium);
        stage_small_gemm(plan, plan.stream_small);
    }

    if (timing) {
        if (plan.total_warps > 0) timing->small_kernel_kind = "warp";
        else timing->small_kernel_kind = nullptr;
    }

    cudaDeviceSynchronize();
    if (timing) { timing->contraction_ms = detail_timing::ms_since(t_stage); }

    // --- Accumulation ---
    t_stage = clock::now();

    stage_accumulate(plan, 0);
    cudaDeviceSynchronize();

    if (timing) { timing->accumulation_ms = detail_timing::ms_since(t_stage); }
    if (timing) { timing->total_ms = detail_timing::ms_since(t_total); }
}

template <typename T>
GeneralSparseResult<T> pipeline_extract_result(PipelinePlan<T>& plan) {
    GeneralSparseResult<T> result;
    result.d_output = plan.d_output;
    result.owns_output = plan.owns_output;
    plan.d_output = nullptr;

    if (plan.num_pairs == 0) {
        result.total_output_elements = 0;
        result.num_output_blocks = 0;
        result.output_rank = plan.output_rank;
        return result;
    }

    result.total_output_elements = plan.accum_map.total_output_elements;
    result.num_output_blocks = plan.accum_map.num_output_blocks;
    result.output_rank = plan.accum_map.output_rank;
    result.output_offsets.resize(plan.accum_map.num_output_blocks);
    result.output_volumes.resize(plan.accum_map.num_output_blocks);
    result.output_block_coords.resize(plan.accum_map.num_output_blocks * plan.accum_map.output_rank);

    for (int64_t i = 0; i < plan.accum_map.num_output_blocks; ++i) {
        result.output_offsets[i] = plan.accum_map.h_output_block_offsets[i];
        result.output_volumes[i] = plan.accum_map.h_output_block_volumes[i];
        for (int32_t d = 0; d < plan.accum_map.output_rank; ++d) {
            result.output_block_coords[i * plan.accum_map.output_rank + d] =
                plan.accum_map.h_output_block_coords[i * plan.accum_map.output_rank + d];
        }
    }

    return result;
}

} // namespace bsptc
