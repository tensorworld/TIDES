#pragma once

#include "pipeline_descs.hpp"
#include "gpu_sparse_input.hpp"
#include "gpu_sparse_device.cuh"
#include "preprocessing_kernel.cuh"
#include "work_partitioner.hpp"
#include "kernels/medium/v2_volume_gemm.cuh"
#include "kernels/small/warp_gemm.cuh"
#include "kernels/small/dot_reduce.cuh"
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
    double output_permute_ms = 0;  // post-contraction permute to einsum output order
    double total_ms = 0;
    // Phase-level timing (plan + prepare + allocate = setup, execute = total_ms)
    double plan_ms = 0;
    double prepare_ms = 0;      // Address-independent shape/geometry build (pipeline_prepare)
    double allocate_ms = 0;     // GPU buffer binding, pointer patching, upload (pipeline_allocate)
    double bind_ms = 0;         // binding/upload subset of allocate (excludes malloc/memset/free)
    double bind_upload_ms = 0;   // upload_reduction_plan
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

// Report a non-fatal pipeline error and return false, so bool-returning
// phase functions can fail in one line: `return pipeline_error("...")`.
// CUDA API failures still throw via check_cuda.
inline bool pipeline_error(const char* msg) {
    std::cerr << "[general_sparse] " << msg << "\n";
    return false;
}

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
    config.rank_overflow = 0;

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
            if (config.num_batch >= MAX_RANK) { config.rank_overflow = 1; continue; }
            config.batch_pos_A[config.num_batch] = pos_A;
            config.batch_pos_B[config.num_batch] = pos_B;
            config.num_batch++;
            internal_order.push_back(c);
        }
    }

    for (char c : index_sets.free_indices_A) {
        int32_t pos_A = find_pos(A_labels, c);
        if (pos_A >= 0) {
            if (config.num_free_A >= MAX_RANK) { config.rank_overflow = 1; continue; }
            config.free_pos_A[config.num_free_A] = pos_A;
            config.num_free_A++;
            internal_order.push_back(c);
        }
    }

    for (char c : index_sets.free_indices_B) {
        int32_t pos_B = find_pos(B_labels, c);
        if (pos_B >= 0) {
            if (config.num_free_B >= MAX_RANK) { config.rank_overflow = 1; continue; }
            config.free_pos_B[config.num_free_B] = pos_B;
            config.num_free_B++;
            internal_order.push_back(c);
        }
    }

    for (char c : index_sets.contracted_indices) {
        int32_t pos_A = find_pos(A_labels, c);
        int32_t pos_B = find_pos(B_labels, c);
        if (pos_A >= 0 && pos_B >= 0) {
            if (config.num_contracted >= MAX_RANK) { config.rank_overflow = 1; continue; }
            config.contracted_pos_A[config.num_contracted] = pos_A;
            config.contracted_pos_B[config.num_contracted] = pos_B;
            config.num_contracted++;
        }
    }

    // Output rank is bounded by MAX_RANK: internal_to_output (and every
    // downstream output-coord array — CoordKeyDynamic, verifier stacks) is
    // a fixed [MAX_RANK] array. Overflowing writes are clamped and flagged;
    // pipeline_plan rejects flagged configs loudly.
    config.output_rank = static_cast<int32_t>(internal_order.size());
    if (config.output_rank > MAX_RANK) config.rank_overflow = 1;
    const int32_t mapped_rank = std::min(config.output_rank, MAX_RANK);

    if (!output_labels.empty()) {
        for (int32_t i = 0; i < mapped_rank; ++i) {
            config.internal_to_output[i] = find_pos_str(output_labels, internal_order[i]);
        }
    } else {
        for (int32_t i = 0; i < mapped_rank; ++i) {
            config.internal_to_output[i] = i;
        }
    }

    return config;
}

// Hypertile transpose kernel configuration — shared between task builder and launcher.
constexpr int HT_TILE_VOL = 1024;   // tile volume (elements per tile)
constexpr int HT_BLOCK_SIZE = 256;  // CUDA threads per block

struct DispatchConfig {
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
// block_offsets/block_volumes are element offsets/counts indexed by block id
// (operand blocks or output blocks — the assembler is layout-agnostic).
inline TransposeFlatMeta assemble_hypertile_batch(
    const std::vector<int>& block_indices,
    const int64_t* block_offsets,
    const int64_t* block_volumes,
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
        if (block_volumes[b] == 0) continue;

        uint32_t shape_id = shape_map.block_to_shape[b];
        if (trivial_shapes[shape_id])
            continue;
        const auto& entry = cache.per_shape[shape_id];

        HypertileMem mem{};
        mem.src_base = block_offsets[b];
        mem.dst_base = pack_offset.empty() ? block_offsets[b] : pack_offset[bi];
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
// Shared execution resources
// ============================================================================

// Streams and events shared by every plan in the process. Creating a CUDA
// stream is cheap, but a LIVE stream is not free: device-wide sync cost
// scales with the number of streams in the context, so per-plan streams
// make cudaDeviceSynchronize O(#live plans) — ~2 ms/sync with ~1000 cached
// plans (measured, DMRG workload). Plans reference this context instead of
// owning resources; executes are serialized per process (the engine's
// existing single-threaded execute contract), so sharing is safe.
struct PipelineStreamContext {
    cudaStream_t stream_large = nullptr;
    cudaStream_t stream_small = nullptr;
    cudaStream_t stream_medium = nullptr;
    cudaStream_t streaming_streams[2] = {};
    cudaEvent_t transpose_done_flat = nullptr;
    cudaEvent_t transpose_done_tiled = nullptr;
    cudaEvent_t gemm_done[2] = {};
    cudaEvent_t copy_done[2] = {};
};

inline PipelineStreamContext& get_stream_context() {
    static PipelineStreamContext ctx = [] {
        PipelineStreamContext c;
        check_cuda(cudaStreamCreate(&c.stream_large), "create stream_large");
        check_cuda(cudaStreamCreate(&c.stream_small), "create stream_small");
        check_cuda(cudaStreamCreate(&c.stream_medium), "create stream_medium");
        check_cuda(cudaEventCreateWithFlags(&c.transpose_done_flat,
                                            cudaEventDisableTiming),
                   "create transpose_done_flat");
        check_cuda(cudaEventCreateWithFlags(&c.transpose_done_tiled,
                                            cudaEventDisableTiming),
                   "create transpose_done_tiled");
        for (int i = 0; i < 2; ++i) {
            check_cuda(cudaStreamCreate(&c.streaming_streams[i]),
                       "create streaming stream");
            check_cuda(cudaEventCreateWithFlags(&c.gemm_done[i],
                                                cudaEventDisableTiming),
                       "create streaming gemm_done event");
            check_cuda(cudaEventCreateWithFlags(&c.copy_done[i],
                                                cudaEventDisableTiming),
                       "create streaming copy_done event");
        }
        return c;
    }();
    return ctx;
}

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

    // Unique shape-pair plan (built in pipeline_plan): per-pair shape-pair id
    // + per-unique-shape-pair geometry/tier. Tier preps use it for dense,
    // hash-free dedup.
    ShapePairPlan shape_pair_plan;

    // Hypertile transpose index (per-operand, built in pipeline_prepare)
    HypertileIndex hypertile_index_A;
    HypertileIndex hypertile_index_B;

    // Per-shape trivial transpose flags (indexed by shape_id, built in pipeline_prepare)
    std::vector<uint8_t> trivial_shapes_A;
    std::vector<uint8_t> trivial_shapes_B;

    // Post-contraction output permute: rearranges each output block from
    // internal order [batch, free_A, free_B] to einsum output label order.
    // Contraction tiers always write internal order into d_output; when the
    // einsum order differs and actually moves data, the permute stage writes
    // the user-facing buffer d_output_final (same block offsets, permuted
    // intra-block layout).
    bool output_permute_needed = false;      // internal_to_output != identity
    bool output_permute_all_trivial = false; // permuted bytes == internal bytes
    bool output_permute_copy_first = false;  // mixed trivial/non-trivial blocks
    BlockShapeMap shape_map_out;             // output-block dims, internal order
    HypertileIndex hypertile_index_out;
    std::vector<uint8_t> trivial_shapes_out;
    TransposeFlatMeta output_permute_meta;
    T* d_output_final = nullptr;
    bool owns_output_final = true;

    // True when the permute stage will physically run.
    bool output_permute_active() const {
        return output_permute_needed && !output_permute_all_trivial;
    }

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
    int32_t* d_dot_indices = nullptr;    // desc indices routed to dot-reduce kernel
    int small_num_dots = 0;

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

    // Set by pipeline_prepare; pipeline_allocate rejects plans where prepare
    // was skipped (bind_small_mems would silently drop small-tier work).
    bool prepare_done = false;

    // Set by pipeline_allocate; pipeline_rebind rejects plans not yet bound.
    bool allocate_done = false;

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
        // Scalar-output (M=N=1, K >= DOT_MIN_K) shapes route to the
        // dot-reduce kernel: zero warps in the prefix, desc index recorded
        // here instead (see dot_reduce.cuh).
        std::vector<uint8_t> shape_is_dot;      // per unique shape
        std::vector<int32_t> h_dot_desc_indices; // expanded desc indices
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
    KernelChoice nonstream_kernels[NUM_SHAPE_QUEUES] = {};

    // Streams — references into the shared PipelineStreamContext (set in
    // pipeline_allocate, never destroyed by the plan)
    cudaStream_t stream_large = nullptr;
    cudaStream_t stream_small = nullptr;
    cudaStream_t stream_medium = nullptr;

    // Streamed large-tier transpose (bounded memory TTGT)
    struct StreamingState {
        // Shared across queues (sequential execution)
        T* d_buf[2] = {};            // double transpose buffers
        CutlassGemmRunnerResult args[2];  // pre-allocated CUTLASS arg buffers
        cudaStream_t streams[2] = {};     // two alternating streams (shared context)
        cudaEvent_t gemm_done[2] = {};    // GEMM completion events (shared context)
        cudaEvent_t copy_done[2] = {};    // H2D args copy completion (shared context)
        int64_t buf_cap = 0;         // elements per buffer
        int64_t max_expanded_problems = 0;
        bool enabled = false;

        // Pre-allocated transpose descriptor buffers (2 slots × 2 operands)
        TransposePrecompBuffers precomp_bufs[2][2];  // [slot][operand: 0=A, 1=B]

        // Per-queue (indexed by ShapeQueue)
        BatchPartitionResult queue_partitions[NUM_SHAPE_QUEUES];
        LargeBlockWork<T> queue_work[NUM_SHAPE_QUEUES];  // final work with ptr_C
        KernelChoice queue_kernels[NUM_SHAPE_QUEUES];

        // Precomputed GEMM + transpose state (first ≤2 batches across all queues).
        // Uses args[0]/args[1] for GEMM, precomp_bufs[slot][op] for transpose.
        // At execute time, only transpose kernels + GEMM launch remain.
        bool precomputed = false;
        int num_precomp = 0;
        TransposeFlatMeta precomp_meta[2][2];  // [batch_slot][operand: 0=A, 1=B]
    };
    StreamingState streaming;

    // Cross-stream sync events for transpose → GEMM ordering (shared context)
    cudaEvent_t transpose_done_flat = nullptr;
    cudaEvent_t transpose_done_tiled = nullptr;

    // Timing
    PipelineTimingBreakdown* timing = nullptr;

    // Measured pipeline allocation (bytes) via cudaMemGetInfo delta.
    // Set in pipeline_allocate; used by populate_timing_stats for accurate peak.
    int64_t measured_alloc_bytes = 0;

    PipelinePlan() = default;
    ~PipelinePlan() {
        // Drain in-flight work first: an exception mid-execute can leave
        // kernels running that still read plan-owned buffers, and pool
        // release (unlike cudaFreeHost) does not synchronize. Normal paths
        // are already idle, so this costs ~µs.
        cudaDeviceSynchronize();
        // Final output: owned until pipeline_extract_result transfers it out
        // (extract nulls d_output; destruction without extraction frees here).
        if (owns_output) cudaFree(d_output);
        // Permuted output buffer: same handoff protocol as d_output.
        if (owns_output_final) cudaFree(d_output_final);
        output_permute_meta.free_gpu();
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
        // streaming.args + nonstream_runners repool via their own destructors
        for (int s = 0; s < 2; ++s)
            for (int op = 0; op < 2; ++op)
                streaming.precomp_bufs[s][op].free_gpu();
        cudaFree(d_small_shapes);
        cudaFree(d_small_mems);
        cudaFree(d_warp_prefix);
        cudaFree(d_dot_indices);
        cudaFree(d_medium_vol_shapes);
        cudaFree(d_medium_vol_mems);
        // reduce_plan and accum_map free themselves (RAII member destructors)
        transpose_precomp_A.free_gpu();
        transpose_precomp_B.free_gpu();
        // Streams/events belong to the shared PipelineStreamContext.
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
    // Product ≤ 3 guarantees at most one non-trivial group, but clamp the
    // scan to written capacity anyway.
    for (int i = 0; i < sq.rank && i < WARP_MAX_DIMS; ++i)
        if (sq.shapes[i] > 1) return static_cast<dim_t>(sq.strides[i]);
    return 0;  // product == 1, stride doesn't matter
}

// Dense (hash-free) dedup of shape_pair_id over one tier's problems: fills
// key_indices (per-problem local unique index) and first_occ (representative
// problem index per local unique); returns the local unique count.
// num_global_shape_pairs bounds the scratch array (shape_pair_plan size).
inline int dedup_shape_pair_ids(const std::vector<GemmProblemDesc>& problems,
                                size_t num_global_shape_pairs,
                                std::vector<uint32_t>& key_indices,
                                std::vector<int>& first_occ)
{
    const int n = static_cast<int>(problems.size());
    key_indices.resize(n);
    first_occ.clear();
    std::vector<int32_t> global_to_local(num_global_shape_pairs, -1);
    int num_unique = 0;
    for (int i = 0; i < n; ++i) {
        int32_t& slot = global_to_local[problems[i].shape_pair_id];
        if (slot < 0) {
            slot = num_unique++;
            first_occ.push_back(i);
        }
        key_indices[i] = static_cast<uint32_t>(slot);
    }
    return num_unique;
}

// Build warp shape geometry for a specific desc type (address-independent).
// Dedup pair shape keys, build WarpShape directly, compute warp prefix.
// Stores result in plan.prepared_small for later bind_small_mems.
// Returns false on squashed-rank overflow — an internal error, since the
// partitioner's rank-eligibility check routes such shapes to CUTLASS.
template <typename DescType, typename T>
bool prepare_small_shapes_impl(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    using ShapeType = typename SmallShapeTemplate<DescType>::ShapeType;
    using LargeFreeType = typename DescType::LargeFreeType;

    const auto& problems = plan.work.small.problems;
    const int n_probs = static_cast<int>(problems.size());
    if (n_probs == 0) return true;

    const auto& config = plan.config;
    const dim_t A_rank = A_gpu.rank;
    const dim_t B_rank = B_gpu.rank;

    // Phase 1: dedup via the shape-pair plan (dense remap, no hashing).
    std::vector<uint32_t> key_indices;
    std::vector<int> first_occ;
    int num_unique = dedup_shape_pair_ids(
        problems, plan.shape_pair_plan.shape_pairs.size(),
        key_indices, first_occ);

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
                if (!sq.fits())
                    return pipeline_error(
                        "prepare_small_shapes: large free dims exceed WARP_MAX_DIMS "
                        "(internal: partitioner should route this shape to CUTLASS)");
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
            if (!sq.fits())
                return pipeline_error(
                    "prepare_small_shapes: contracted dims exceed WARP_MAX_DIMS "
                    "(internal: partitioner should route this shape to CUTLASS)");
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
    // Scalar-output shapes (M*N == 1) with K >= DOT_MIN_K route to the
    // dot-reduce kernel: zero warps here, desc indices collected below.
    ps.shape_strides.resize(num_unique);
    ps.shape_is_dot.assign(static_cast<size_t>(num_unique), 0);
    for (int u = 0; u < num_unique; ++u) {
        const auto& shp = templates[u].shape;
        const bool is_dot = (shp.small_size * shp.large_size == 1) &&
                            (shp.K_total >= DOT_MIN_K);
        ps.shape_is_dot[u] = is_dot ? 1 : 0;
        ps.shape_strides[u].batch_stride_a = templates[u].batch_stride_a;
        ps.shape_strides[u].batch_stride_b = templates[u].batch_stride_b;
        ps.shape_strides[u].batch_stride_c = templates[u].batch_stride_c;
        ps.shape_strides[u].warps_per = is_dot
            ? 0
            : (shp.small_size * shp.large_size + 31) / 32;
    }
    ps.key_indices = std::move(key_indices);

    // Count total expanded descs and build warp prefix (address-independent).
    int total_descs = 0;
    for (int p = 0; p < n_probs; ++p)
        total_descs += problems[p].batch_size;
    ps.total_descs = total_descs;

    ps.h_dot_desc_indices.clear();
    if (total_descs > 0) {
        ps.h_warp_prefix.resize(total_descs + 1);
        ps.h_warp_prefix[0] = 0;
        int idx = 0;
        for (int p = 0; p < n_probs; ++p) {
            const uint32_t u = ps.key_indices[p];
            const int32_t warps = ps.shape_strides[u].warps_per;
            const bool dot = ps.shape_is_dot[u] != 0;
            for (dim_t b = 0; b < problems[p].batch_size; ++b, ++idx) {
                ps.h_warp_prefix[idx + 1] = ps.h_warp_prefix[idx] + warps;
                if (dot) ps.h_dot_desc_indices.push_back(idx);
            }
        }
        ps.total_warps = ps.h_warp_prefix[total_descs];
    }

    plan.small_num_descs = total_descs;
    plan.small_num_shapes = num_unique;
    plan.total_warps = ps.total_warps;
    plan.small_num_dots = static_cast<int>(ps.h_dot_desc_indices.size());
    return true;
}

// Build warp shape geometry (address-independent, no GPU alloc).
// Dispatches to prepare_small_shapes_impl based on dim interleaving.
// INT_LARGE = true if either operand has multi-dim free dims (conservative:
// the large dim might sometimes be the simple one, in which case IntDim
// is filled with num_dims=1 — a trivial single-multiply decode).
template <typename T>
bool prepare_small_shapes(
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
    bool ok = false;
    switch (desc_type) {
        case 0: ok = prepare_small_shapes_impl<DescSS>(plan, A_gpu, B_gpu); break;
        case 1: ok = prepare_small_shapes_impl<DescSI>(plan, A_gpu, B_gpu); break;
        case 2: ok = prepare_small_shapes_impl<DescIS>(plan, A_gpu, B_gpu); break;
        case 3: ok = prepare_small_shapes_impl<DescII>(plan, A_gpu, B_gpu); break;
    }
    if (!ok) return false;

    // Use squash-derived contiguity from prepared state.
    // The original position-based check can be wrong when squash eliminates
    // size-1 contracted dims, changing which dim is innermost.
    bool contig_k_a = plan.prepared_small.contig_k_a;
    bool contig_k_b = plan.prepared_small.contig_k_b;

    plan.warp_variant = (int_large ? 8 : 0) | (int_k ? 4 : 0) |
                        (contig_k_a ? 2 : 0) | (contig_k_b ? 1 : 0);
    return true;
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

    // Upload shapes/mems/prefix. Buffers are sized by prepare-time metadata,
    // which is fixed for the plan's lifetime — on a rebind (pipeline_rebind)
    // they already exist and only the contents are re-uploaded.
    if (!plan.d_small_shapes)
        check_cuda(cudaMalloc(&plan.d_small_shapes, ps.h_shape_bytes.size()),
                   "malloc warp_shapes");
    check_cuda(cudaMemcpyAsync(plan.d_small_shapes, ps.h_shape_bytes.data(),
                               ps.h_shape_bytes.size(),
                               cudaMemcpyHostToDevice, plan.stream_small),
               "memcpy warp_shapes");

    if (!plan.d_small_mems)
        check_cuda(cudaMalloc(&plan.d_small_mems,
                              ps.total_descs * sizeof(WarpMem)), "malloc warp_mems");
    check_cuda(cudaMemcpyAsync(plan.d_small_mems, h_mems.data(),
                               ps.total_descs * sizeof(WarpMem),
                               cudaMemcpyHostToDevice, plan.stream_small),
               "memcpy warp_mems");

    if (!plan.d_warp_prefix)
        check_cuda(cudaMalloc(&plan.d_warp_prefix,
                              (ps.total_descs + 1) * sizeof(int32_t)), "malloc warp_prefix");
    check_cuda(cudaMemcpyAsync(plan.d_warp_prefix, ps.h_warp_prefix.data(),
                               (ps.total_descs + 1) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, plan.stream_small),
               "memcpy warp_prefix");

    if (!ps.h_dot_desc_indices.empty()) {
        const size_t dot_bytes =
            ps.h_dot_desc_indices.size() * sizeof(int32_t);
        if (!plan.d_dot_indices)
            check_cuda(cudaMalloc(&plan.d_dot_indices, dot_bytes),
                       "malloc dot_indices");
        check_cuda(cudaMemcpyAsync(plan.d_dot_indices,
                                   ps.h_dot_desc_indices.data(), dot_bytes,
                                   cudaMemcpyHostToDevice, plan.stream_small),
                   "memcpy dot_indices");
    }
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

    // Squash both dim groups, then check the combined rank BEFORE writing any
    // arrays. On overflow, return a sentinel rank > VOL_MAX_RANK — callers
    // (prepare_medium_shapes, volume_rank_eligible) reject such shapes; the
    // partitioner routes them to CUTLASS.
    auto sqk = layout::squash_dimensions<VOL_MAX_RANK>(k_tile_shapes, k_strides, k_rank);

    int32_t free_shapes[MAX_RANK];
    int64_t free_strides[MAX_RANK];
    for (int i = 0; i < num_free; ++i) {
        free_shapes[i] = static_cast<int32_t>(block_dims[free_pos[i]]);
        free_strides[i] = compute_dim_stride(block_dims, block_rank, free_pos[i]);
    }
    auto sqf = layout::squash_dimensions<VOL_MAX_RANK>(free_shapes, free_strides, num_free);

    OperandLoadInfo out{};
    if (!sqk.fits() || !sqf.fits() || sqk.rank + sqf.rank > VOL_MAX_RANK) {
        out.rank = VOL_MAX_RANK + 1;  // sentinel: volume-ineligible
        out.tile_volume = 0;
        return out;
    }

    // K tile dims first (innermost). Divmod chain needs innermost first
    // (row-major convention), so we reverse.
    for (int i = sqk.rank - 1; i >= 0; --i) {
        entries[n].shape = static_cast<int32_t>(sqk.shapes[i]);
        entries[n].stride = sqk.strides[i];
        n++;
    }

    // Free dims next (outermost) — reversed for the divmod chain.
    for (int i = sqf.rank - 1; i >= 0; --i) {
        entries[n].shape = static_cast<int32_t>(sqf.shapes[i]);
        entries[n].stride = sqf.strides[i];
        n++;
    }

    // Degenerate safety: all dims were size 1
    if (n == 0) { entries[0] = {1, 0}; n = 1; }

    // Build divmod chain: divmod[0] is innermost (first K dim)
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

// ---- Partition-time rank-eligibility checks --------------------------------
// The warp and volume desc formats hold a bounded number of squashed dims
// (WARP_MAX_DIMS / VOL_MAX_RANK). Shapes whose squashed rank exceeds those
// capacities must be routed to CUTLASS at partition time; the desc builders
// treat overflow as an internal error. Squashing never increases rank, so
// pipeline_plan only consults these (memoized per unique shape pair) when the
// config's raw dim counts could overflow — the common case costs nothing.

// True iff the warp-tier desc format can represent this shape pair:
// squashed large-side free rank and dual-squashed K rank ≤ WARP_MAX_DIMS.
// (The small side has product ≤ 3 → at most one non-trivial dim, always safe.)
inline bool warp_rank_eligible(
    const ContractionConfig& config,
    const dim_t* A_dims, dim_t A_rank,
    const dim_t* B_dims, dim_t B_rank,
    int64_t M, int64_t N)
{
    // Mirror prepare_small_shapes_impl: the larger free dim uses IntDim.
    const bool m_small = (M <= N);
    const dim_t* large_dims = m_small ? B_dims : A_dims;
    dim_t large_rank = m_small ? B_rank : A_rank;
    const int32_t* large_pos = m_small ? config.free_pos_B : config.free_pos_A;
    int32_t num_large = m_small ? config.num_free_B : config.num_free_A;

    if (num_large > 1) {
        int32_t f_shapes[MAX_RANK];
        int64_t f_strides[MAX_RANK];
        for (int32_t fi = 0; fi < num_large; ++fi) {
            int32_t pos = large_pos[fi];
            f_shapes[fi] = static_cast<int32_t>(large_dims[pos]);
            f_strides[fi] = compute_dim_stride(large_dims, large_rank, pos);
        }
        if (!layout::squash_dimensions<WARP_MAX_DIMS>(
                f_shapes, f_strides, num_large).fits())
            return false;
    }

    if (config.num_contracted > 1) {
        int32_t k_shapes[MAX_RANK];
        int64_t k_strides_a[MAX_RANK], k_strides_b[MAX_RANK];
        for (int32_t ci = 0; ci < config.num_contracted; ++ci) {
            int32_t pos_a = config.contracted_pos_A[ci];
            int32_t pos_b = config.contracted_pos_B[ci];
            k_shapes[ci] = static_cast<int32_t>(A_dims[pos_a]);
            k_strides_a[ci] = compute_dim_stride(A_dims, A_rank, pos_a);
            k_strides_b[ci] = compute_dim_stride(B_dims, B_rank, pos_b);
        }
        if (!layout::squash_dimensions_dual<WARP_MAX_DIMS>(
                k_shapes, k_strides_a, k_strides_b, config.num_contracted).fits())
            return false;
    }
    return true;
}

// True iff the volume-tier desc format can represent this shape pair:
// combined squashed K+free rank ≤ VOL_MAX_RANK for both operands.
inline bool volume_rank_eligible(
    const ContractionConfig& config,
    const dim_t* A_dims, dim_t A_rank,
    const dim_t* B_dims, dim_t B_rank)
{
    KDimInfo k_dims[MAX_RANK];
    int num_k = build_k_dims(config, A_dims, A_rank, B_dims, B_rank, k_dims);

    int32_t k_shapes[MAX_RANK];
    int64_t k_strides_A[MAX_RANK], k_strides_B[MAX_RANK];
    for (int d = 0; d < num_k; ++d) {
        k_shapes[d] = k_dims[d].dim;
        k_strides_A[d] = k_dims[d].stride_a;
        k_strides_B[d] = k_dims[d].stride_b;
    }

    OperandLoadInfo A_info = build_operand_load_info(
        config.free_pos_A, config.num_free_A,
        k_shapes, k_strides_A, num_k, A_dims, A_rank);
    if (A_info.rank > VOL_MAX_RANK) return false;
    OperandLoadInfo B_info = build_operand_load_info(
        config.free_pos_B, config.num_free_B,
        k_shapes, k_strides_B, num_k, B_dims, B_rank);
    return B_info.rank <= VOL_MAX_RANK;
}

// Build the unique shape-pair plan: dedup (shape_A, shape_B) across pairs and
// compute GEMM geometry + tier ONCE per unique shape pair. Per-pair cost in
// partition_by_threshold drops to a table lookup, and the tier preps reuse
// shape_pair_id for hash-free dedup.
// Tier rules:
//   warp:   min(M,N) < MIN_CUTLASS_DIM && max(M,N) <= MAX_WARP_DIM
//   volume: vol_tile_count <= tile_limit && M*K + N*K <= vol_threshold
//   CUTLASS otherwise. Warp/volume additionally require the squashed-rank
//   eligibility checks (their desc formats hold a bounded number of dims);
//   squash never increases rank, so configs whose raw dim counts fit skip
//   those checks entirely.
template <typename T>
ShapePairPlan build_shape_pair_plan(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const std::vector<BlockPairInfo>& pairs,
    const ContractionConfig& config,
    const BlockShapeMap& shape_map_A,
    const BlockShapeMap& shape_map_B,
    GemmMode gemm_mode,
    bool needs_transpose,
    int vol_max_tiles_override)
{
    ShapePairPlan sp;
    const int n = static_cast<int>(pairs.size());
    if (n == 0) return sp;

    std::vector<uint64_t> keys(n);
    for (int i = 0; i < n; ++i)
        keys[i] = make_pair_shape_key(
            shape_map_A.block_to_shape[pairs[i].block_idx_A],
            shape_map_B.block_to_shape[pairs[i].block_idx_B]);
    sp.pair_shape_id.resize(n);
    std::vector<int> first_occ(n);
    const int num_unique = pair_key_dedup(keys.data(), n,
                                          sp.pair_shape_id.data(),
                                          first_occ.data());

    // Volume shape gate: tiles ≤ 256 (8-warp CTA) when transpose is needed,
    // tiles ≤ 64 (2-warp CTA) otherwise.  Override via config.
    const int vol_tile_limit = (vol_max_tiles_override > 0)
        ? vol_max_tiles_override
        : (needs_transpose ? VOL_MAX_TILES : VOL_MAX_TILES_NO_TRANSPOSE);

    const bool warp_may_overflow =
        std::max(config.num_free_A, config.num_free_B) > WARP_MAX_DIMS ||
        config.num_contracted > WARP_MAX_DIMS;
    const bool vol_may_overflow =
        std::max(config.num_free_A, 1) + std::max(config.num_contracted, 1) >
            VOL_MAX_RANK ||
        std::max(config.num_free_B, 1) + std::max(config.num_contracted, 1) >
            VOL_MAX_RANK;

    sp.shape_pairs.resize(num_unique);
    for (int u = 0; u < num_unique; ++u) {
        const BlockPairInfo& pair = pairs[first_occ[u]];
        const dim_t* A_dims = shape_map_A.shape_dims(
            shape_map_A.block_to_shape[pair.block_idx_A]);
        const dim_t* B_dims = shape_map_B.shape_dims(
            shape_map_B.block_to_shape[pair.block_idx_B]);

        ShapePairInfo& info = sp.shape_pairs[u];
        compute_gemm_dimensions(A_dims, B_dims, A_gpu.rank, B_gpu.rank,
                                /*A_blk_idx=*/0, /*B_blk_idx=*/0, config,
                                info.M, info.N, info.K, info.batch_size);
        info.lda = gemm_mode_a_transposed(gemm_mode) ? info.M : info.K;
        info.ldb = gemm_mode_b_transposed(gemm_mode) ? info.K : info.N;
        info.ldc = info.N;

        if (std::min(info.M, info.N) < MIN_CUTLASS_DIM &&
            std::max(info.M, info.N) <= MAX_WARP_DIM &&
            (!warp_may_overflow ||
             warp_rank_eligible(config, A_dims, A_gpu.rank, B_dims, B_gpu.rank,
                                info.M, info.N))) {
            info.tier = 0;
        } else if (vol_tile_count(info.M, info.N) <= vol_tile_limit &&
                   (static_cast<int64_t>(info.M) * info.K +
                    static_cast<int64_t>(info.N) * info.K) <= vol_threshold<T>() &&
                   (!vol_may_overflow ||
                    volume_rank_eligible(config, A_dims, A_gpu.rank,
                                         B_dims, B_gpu.rank))) {
            info.tier = 1;
        } else {
            info.tier = 2;
        }
    }

    // Diagnostic dump: BSPTC_DUMP_SHAPE_PLAN=<min_K> prints every unique
    // shape pair with K >= min_K to stderr (one line per unique shape,
    // with the number of block pairs mapping to it).
    static const long dump_min_k = [] {
        const char* s = std::getenv("BSPTC_DUMP_SHAPE_PLAN");
        return s && *s ? std::strtol(s, nullptr, 10) : -1;
    }();
    if (dump_min_k >= 0) {
        std::vector<int> uses(static_cast<size_t>(num_unique), 0);
        for (int i = 0; i < n; ++i) uses[sp.pair_shape_id[i]]++;
        for (int u = 0; u < num_unique; ++u) {
            const ShapePairInfo& info = sp.shape_pairs[u];
            if (info.K < dump_min_k) continue;
            fprintf(stderr,
                    "[shape_plan] tier=%d M=%d N=%d K=%d batch=%d pairs=%d\n",
                    static_cast<int>(info.tier), info.M, info.N, info.K,
                    info.batch_size, uses[u]);
        }
    }
    return sp;
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

    // Single pass over problems (a multi-MB stream at scale): dedup shape
    // pairs via dense remap of shape_pair_id (no hashing), sum batch sizes,
    // and count problems per output block.
    const int64_t nob = plan.accum_map.num_output_blocks;
    std::vector<int32_t> ob_count(static_cast<size_t>(nob), 0);
    std::vector<uint32_t> key_indices(n_probs);
    std::vector<int> first_occ;
    std::vector<int32_t> global_to_local(
        plan.shape_pair_plan.shape_pairs.size(), -1);
    int num_unique = 0;
    int total = 0;
    for (int i = 0; i < n_probs; ++i) {
        int32_t& slot = global_to_local[problems[i].shape_pair_id];
        if (slot < 0) {
            slot = num_unique++;
            first_occ.push_back(i);
        }
        key_indices[i] = static_cast<uint32_t>(slot);
        total += problems[i].batch_size;
        ob_count[static_cast<size_t>(
            plan.accum_map.h_pair_output_block_idx[problems[i].pair_idx])]++;
    }

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
    pm.total_descs = total;
    pm.max_smem_elems = max_smem;
    pm.max_tiles = max_tc;

    plan.medium_vol_num_shapes = pm.num_shapes;
    plan.medium_vol_max_smem_elems = pm.max_smem_elems;
    plan.medium_vol_max_tiles = pm.max_tiles;

    // Group medium problems by output block (CSR flat arrays; counts come
    // from the fused pass above). bind_medium_mems iterates in this order
    // to produce C-pointer-sorted mems without an O(n log n) sort.
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
    if (pm.key_indices.size() != static_cast<size_t>(n_probs))
        return pipeline_error("bind_medium_mems: pipeline_prepare not called");

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
        // Upload shapes/mems. Sizes are fixed by prepare-time metadata — on a
        // rebind (pipeline_rebind) the buffers exist and are re-filled in place.
        if (!plan.d_medium_vol_shapes)
            check_cuda(malloc_device_cached(
                           reinterpret_cast<void**>(&plan.d_medium_vol_shapes),
                           pm.h_shapes.size() * sizeof(VolumeShape)),
                       "malloc d_medium_vol_shapes");
        check_cuda(cudaMemcpyAsync(plan.d_medium_vol_shapes, pm.h_shapes.data(),
                                    pm.h_shapes.size() * sizeof(VolumeShape),
                                    cudaMemcpyHostToDevice, plan.stream_medium),
                   "memcpy d_medium_vol_shapes");

        if (!plan.d_medium_vol_mems)
            check_cuda(malloc_device_cached(
                           reinterpret_cast<void**>(&plan.d_medium_vol_mems),
                           h_mems.size() * sizeof(VolumeMem)),
                       "malloc d_medium_vol_mems");
        check_cuda(cudaMemcpyAsync(plan.d_medium_vol_mems, h_mems.data(),
                                    h_mems.size() * sizeof(VolumeMem),
                                    cudaMemcpyHostToDevice, plan.stream_medium),
                   "memcpy d_medium_vol_mems");
    }
    return true;
}

// --- pipeline_plan step: early streaming partition --------------------------
// Classifies CUTLASS work into shape queues and partitions each queue into
// streaming batches. Pairs that partitioning separates out are rerouted to
// d_output (pair_tier 0) and recorded in separated_pair_indices.
// Returns whether streaming is enabled (transpose_buf_cap > 0 and large work).
template <typename T>
bool plan_streaming_separation(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DispatchConfig& dispatch_config,
    std::vector<uint8_t>& pair_tier,
    std::vector<size_t>& separated_pair_indices)
{
    if (dispatch_config.transpose_buf_cap <= 0 || plan.work.large.empty())
        return false;

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
    return true;
}

// --- pipeline_plan step: temp-buffer sizing + per-pair offsets ---------------
// Detects direct-write CUTLASS pairs (sole temp-tier writer of a non-mixed
// output block, which can bypass the temp buffer), then assigns temp-buffer
// offsets to the rest: streaming reuses slots across (queue, batch) groups
// (serialized batches accumulate via CUTLASS beta=1), non-streaming packs
// each pair into its own slot. Sets pairs_host[].output_offset and
// plan.total_pair_output_size.
template <typename T>
void plan_temp_buffer_offsets(
    PipelinePlan<T>& plan,
    const std::vector<uint8_t>& pair_tier,
    const std::vector<uint8_t>& is_direct,
    bool early_can_stream)
{
    const size_t np = static_cast<size_t>(plan.num_pairs);

    int64_t compact_offset = 0;
    if (early_can_stream) {
        // Streaming: reuse temp slots across batches for the same output block.
        // Pairs in different (queue, batch) groups are serialized, so they can
        // share a temp slot and accumulate via CUTLASS beta=1.

        // Build pair_idx → (queue << 32 | batch) concurrent-group key; the
        // packed form cannot collide across queues at any batch count.
        // Flat array indexed by pair (-1 = not in any batch); avoids hash-map
        // churn on the O(num_pairs) hot path.
        std::vector<int64_t> pair_qb(np, -1);
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            const auto& qwork = plan.staging_classified.queues[qi];
            const auto& qpart = plan.streaming.queue_partitions[qi];
            for (int bi = 0; bi < static_cast<int>(qpart.batches.size()); ++bi) {
                int64_t key = (static_cast<int64_t>(qi) << 32) |
                              static_cast<uint32_t>(bi);
                for (int li : qpart.batches[bi].problem_indices) {
                    pair_qb[static_cast<size_t>(qwork.problems[li].pair_idx)] = key;
                }
            }
        }

        // Per output block: find max concurrent pairs, assign shared offsets.
        // Scratch vectors reused across blocks; groups per block are few, so
        // a linear scan over (key, count) beats a hash map.
        std::vector<size_t> ob_pairs;
        std::vector<std::pair<int64_t, int>> group_slots;  // (qb key, next slot)
        for (int64_t ob = 0; ob < plan.accum_map.num_output_blocks; ++ob) {
            const int64_t begin = plan.accum_map.h_output_pair_offsets[ob];
            const int64_t end = plan.accum_map.h_output_pair_offsets[ob + 1];

            // Collect non-direct CUTLASS pairs for this output block
            ob_pairs.clear();
            for (int64_t i = begin; i < end; ++i) {
                size_t pi = static_cast<size_t>(plan.accum_map.h_output_pair_indices[i]);
                if (pair_tier[pi] != 0 && !is_direct[pi])
                    ob_pairs.push_back(pi);
            }
            if (ob_pairs.empty()) continue;

            int64_t volume = plan.pairs_host[ob_pairs[0]].output_volume;

            // Count pairs per concurrent group; find max
            group_slots.clear();
            int max_slots = 0;
            for (size_t pi : ob_pairs) {
                const int64_t key = pair_qb[pi];
                int cnt = 0;
                for (auto& [k, c] : group_slots) {
                    if (k == key) { cnt = ++c; break; }
                }
                if (cnt == 0) { group_slots.emplace_back(key, 1); cnt = 1; }
                max_slots = std::max(max_slots, cnt);
            }

            // Assign offsets: pairs in same (queue, batch) get sequential slots
            int64_t block_base = compact_offset;
            compact_offset += static_cast<int64_t>(max_slots) * volume;

            for (auto& [k, c] : group_slots) c = 0;  // reuse as next-slot cursor
            for (size_t pi : ob_pairs) {
                const int64_t key = pair_qb[pi];
                int slot = 0;
                for (auto& [k, c] : group_slots) {
                    if (k == key) { slot = c++; break; }
                }
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
}

// ============================================================================
// Phase 1: CPU planning + small GPU allocs. No big buffer allocation.
// After this returns, call pipeline_prepare() to build shape/geometry state,
// pipeline_query_workspace() to get buffer sizes, then pipeline_allocate()
// to bind buffers and finalize GPU setup. Skipping pipeline_prepare is
// rejected by pipeline_allocate.
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
    plan.prepare_done = false;  // re-planning invalidates any prior prepare

    plan.config = build_contraction_config(
        index_sets, A_gpu.labels, B_gpu.labels, output_labels);
    plan.output_rank = plan.config.output_rank;

    // Fixed-capacity guard: ContractionConfig arrays (and every downstream
    // output-coord array) hold at most MAX_RANK entries. build_contraction_config
    // clamps overflowing writes and sets rank_overflow; reject here before any
    // clamped (unusable) config reaches preprocessing.
    if (plan.config.rank_overflow)
        return pipeline_error(
            "pipeline_plan: contraction rank exceeds MAX_RANK "
            "(output rank or per-category dim count too large)");

    // Post-contraction permute needed when the einsum output label order
    // differs from internal order [batch, free_A, free_B].
    plan.output_permute_needed = false;
    for (int32_t d = 0; d < plan.config.output_rank; ++d)
        if (plan.config.internal_to_output[d] != d)
            plan.output_permute_needed = true;
    if (plan.output_permute_needed && dispatch_config.output_beta != 0)
        return pipeline_error(
            "pipeline_plan: output_beta != 0 requires the einsum output order "
            "to match internal order (accumulating onto a permuted output "
            "buffer is not supported)");

    // Enforce batch-leftmost constraint: batch dimensions must occupy positions
    // 0..num_batch-1 in both operands. This guarantees that volume and warp
    // kernels can handle interleaved layouts natively without pre-transpose.
    if (plan.config.num_batch > 0) {
        if (!batch_dims_are_leading(plan.config.batch_pos_A, plan.config.num_batch))
            return pipeline_error("pipeline_plan: batch dims not leading in A");
        if (!batch_dims_are_leading(plan.config.batch_pos_B, plan.config.num_batch))
            return pipeline_error("pipeline_plan: batch dims not leading in B");
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

    // Per-tensor block shape dedup: blocks with identical dims share a shape ID.
    // Built before partitioning: the rank-eligibility memo below is keyed by
    // pair shape ID, and downstream desc setup reuses the maps to compute
    // geometry once per unique shape rather than once per block.
    plan.shape_map_A = build_block_shape_map(
        A_gpu.block_dims.data(), A_gpu.num_blocks, A_gpu.rank);
    plan.shape_map_B = build_block_shape_map(
        B_gpu.block_dims.data(), B_gpu.num_blocks, B_gpu.rank);

    // Unique shape-pair plan: geometry, rank eligibility, and tier computed
    // once per unique (shape_A, shape_B) pair.
    plan.shape_pair_plan = build_shape_pair_plan(
        A_gpu, B_gpu, plan.pairs_host, plan.config,
        plan.shape_map_A, plan.shape_map_B, plan.gemm_mode,
        /*needs_transpose=*/physical_transpose_needed,
        dispatch_config.vol_max_tiles);

    // Single classification pass — pure assembly from the shape-pair plan.
    plan.work = partition_by_threshold(
        A_gpu, B_gpu, plan.pairs_host, plan.shape_pair_plan, plan.gemm_mode);

    // work_tier: 0 = warp, 1 = volume, 2 = CUTLASS (for AccumulationMap flags),
    // read straight from the shape-pair plan (no per-tier rebuild loops).
    std::vector<uint8_t> work_tier(static_cast<size_t>(plan.num_pairs));
    for (size_t p = 0; p < work_tier.size(); ++p)
        work_tier[p] = plan.shape_pair_plan
                           .shape_pairs[plan.shape_pair_plan.pair_shape_id[p]].tier;

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

    // Build pair_tier: 0 = small/volume (writes directly to d_output with atomics),
    //                  2 = large/CUTLASS (writes to temp buffer, eligible for direct-write)
    // Volume tier (medium) writes directly to d_output like warp, so pair_tier = 0.
    // Derived from work_tier: large tier (2) is the default there.
    std::vector<uint8_t> pair_tier(static_cast<size_t>(plan.num_pairs));
    for (size_t p = 0; p < pair_tier.size(); ++p)
        pair_tier[p] = (work_tier[p] == 2) ? 2 : 0;

    // Early streaming partition: needed before temp buffer sizing so that
    // separated pairs can be routed to d_output (skipping temp buffer).
    std::vector<size_t> separated_pair_indices;
    const bool early_can_stream = plan_streaming_separation(
        plan, A_gpu, B_gpu, dispatch_config, pair_tier, separated_pair_indices);

    // Direct-write eligibility, computed once (after streaming separation
    // finalizes pair_tier) for both temp-offset assignment and the reduction
    // plan — these previously each recomputed the same singleton scan.
    const std::vector<uint8_t> is_direct =
        compute_direct_write_eligibility(plan.accum_map, pair_tier);

    // Right-size temp buffer: only non-direct-write CUTLASS pairs need space.
    plan_temp_buffer_offsets(plan, pair_tier, is_direct, early_can_stream);

    // Reduction plan (host-side only; GPU upload deferred to pipeline_allocate)
    plan.reduce_plan = build_reduction_plan(plan.accum_map, pair_tier, is_direct,
                                            plan.pairs_host.data());
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

    // Output permute: the internal-order accumulator becomes a workspace
    // buffer (the user-facing d_output holds the einsum-order result).
    // Requires pipeline_prepare (trivial-shape info) to have run.
    if (plan.output_permute_active())
        ws += align_up(plan.accum_map.total_output_elements * elem, ALIGN);

    // One-time slack for aligning an arbitrary workspace base: take() aligns
    // the base to 256 bytes and charges the padding against the budget. CUDA
    // allocations are already 256-aligned, so this only matters for
    // sub-allocated (offset) bases — but exact sizing must work for any base.
    info.workspace_bytes = ws > 0 ? ws + ALIGN - 1 : 0;
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
        }
    }
    bw.num_gemm_problems = static_cast<int64_t>(bw.ptr_A.size());
    return bw;
}

// ============================================================================
// Phase 2: Address-independent shape/geometry build. No GPU pointers needed.
// Builds shapes, dedup, tile selection, hypertile index — everything that
// can be reused across GPU buffer reallocations.
// Returns false on failure (reported via pipeline_error); prepare_done is set
// only on success, so a failed prepare is caught again by pipeline_allocate.
// ============================================================================
template <typename T>
bool pipeline_prepare(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    if (plan.num_pairs == 0) {
        plan.prepare_done = true;
        return true;
    }

    // Small tier: shape geometry + warp prefix
    if (!plan.work.small.empty()) {
        if (!prepare_small_shapes(plan, A_gpu, B_gpu))
            return false;  // rank overflow already reported via pipeline_error
    }

    // Medium tier: VolumeShape + divmod
    if (!plan.work.medium.empty()) {
        if (!prepare_medium_shapes(plan, A_gpu, B_gpu)) {
            return pipeline_error(
                "pipeline_prepare: prepare_medium_shapes failed "
                "(internal: partitioner rank-eligibility should prevent this)");
        }
    }

    // Large tier: classify shapes + select tiles
    if (!plan.work.large.empty()) {
        if (!plan.streaming.enabled) {
            plan.nonstream_classified = classify_cutlass_work(plan.work);
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
                if (plan.nonstream_classified.queues[qi].empty()) continue;
                plan.nonstream_kernels[qi] = best_kernel_for_queue(
                    plan.nonstream_classified.queues[qi]);
            }
        } else {
            // Streaming: compute tiles from staging_classified (built in pipeline_plan)
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
                if (plan.staging_classified.queues[qi].empty()) continue;
                plan.streaming.queue_kernels[qi] = best_kernel_for_queue(
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
            if (plan.hypertile_index_A.rank_overflow)
                return pipeline_error(
                    "pipeline_prepare: A transpose shape exceeds HT_MAX_RANK "
                    "(layout not representable by the hypertile kernel)");
        }
        if (plan.layout_plan.plan_B.transpose &&
            plan.layout_plan.plan_B.dispatch_kind == TransposeDispatchKind::GenericPermute) {
            plan.hypertile_index_B = build_hypertile_index<T>(
                B_gpu, plan.layout_plan.plan_B, plan.shape_map_B);
            if (plan.hypertile_index_B.rank_overflow)
                return pipeline_error(
                    "pipeline_prepare: B transpose shape exceeds HT_MAX_RANK "
                    "(layout not representable by the hypertile kernel)");
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

    // Output permute: shape dedup + hypertile index over output blocks.
    // Block dims (internal order) come from the section sizes of the source
    // dims at each block coordinate.
    if (plan.output_permute_needed) {
        const auto& am = plan.accum_map;
        const int32_t rank = am.output_rank;
        const auto& config = plan.config;

        auto section_size = [](const auto& gpu, int32_t dim_pos, int32_t blk) {
            return gpu.block_sizes_flat[gpu.block_sizes_offsets[dim_pos] + blk];
        };

        std::vector<dim_t> out_dims(
            static_cast<size_t>(am.num_output_blocks) * rank);
        for (int64_t b = 0; b < am.num_output_blocks; ++b) {
            const int32_t* coord = &am.h_output_block_coords[b * rank];
            dim_t* dims = &out_dims[static_cast<size_t>(b) * rank];
            int32_t pos = 0;
            for (int32_t i = 0; i < config.num_batch; ++i, ++pos)
                dims[pos] = section_size(A_gpu, config.batch_pos_A[i], coord[pos]);
            for (int32_t i = 0; i < config.num_free_A; ++i, ++pos)
                dims[pos] = section_size(A_gpu, config.free_pos_A[i], coord[pos]);
            for (int32_t i = 0; i < config.num_free_B; ++i, ++pos)
                dims[pos] = section_size(B_gpu, config.free_pos_B[i], coord[pos]);
        }

        plan.shape_map_out = build_block_shape_map(
            out_dims.data(), static_cast<int>(am.num_output_blocks), rank);
        plan.hypertile_index_out = build_hypertile_index_perm(
            plan.shape_map_out, rank, config.internal_to_output);
        if (plan.hypertile_index_out.rank_overflow)
            return pipeline_error(
                "pipeline_prepare: output permute exceeds HT_MAX_RANK "
                "(einsum output layout not representable by the hypertile kernel)");
        plan.trivial_shapes_out = compute_trivial_shapes(
            plan.shape_map_out,
            plan.hypertile_index_out.valid ? &plan.hypertile_index_out : nullptr);

        // All-trivial: the permuted buffer would be bit-identical to the
        // internal one (only size-1 dims move) — no stage, no extra buffer.
        // Mixed: trivial blocks are covered by a full-buffer device copy
        // before the permute kernel overwrites the non-trivial regions
        // (rare; avoids a per-block copy kernel).
        bool all_trivial = true, any_trivial = false;
        for (uint8_t t : plan.trivial_shapes_out) {
            if (t) any_trivial = true; else all_trivial = false;
        }
        plan.output_permute_all_trivial = all_trivial;
        plan.output_permute_copy_first = !all_trivial && any_trivial;
    }

    // Set only on success: pipeline_allocate rejects plans where preparation
    // was skipped OR failed partway.
    plan.prepare_done = true;
    return true;
}

// Bump allocator over a user-provided workspace (256-byte aligned).
// When no workspace is provided, take() returns nullptr and callers
// self-allocate via malloc_device_cached. A workspace smaller than
// pipeline_query_workspace() reported is a caller error: take() throws
// rather than handing out out-of-bounds device pointers.
struct WorkspaceBump {
    void* ws = nullptr;
    int64_t bytes = 0;

    bool provided() const { return ws != nullptr; }

    void* take(int64_t n) {
        if (!ws) return nullptr;
        constexpr uintptr_t ALIGN = 256;
        uintptr_t p = reinterpret_cast<uintptr_t>(ws);
        p = (p + ALIGN - 1) & ~(ALIGN - 1);
        int64_t used = static_cast<int64_t>(p - reinterpret_cast<uintptr_t>(ws)) + n;
        if (used > bytes)
            throw std::runtime_error(
                "pipeline_allocate: user workspace too small "
                "(size it with pipeline_query_workspace)");
        void* result = reinterpret_cast<void*>(p);
        ws = reinterpret_cast<void*>(p + n);
        bytes -= used;
        return result;
    }
};

// --- pipeline_allocate step: bind or allocate d_output + temp buffer --------
template <typename T>
void allocate_bind_output(PipelinePlan<T>& plan, T* d_output,
                          WorkspaceBump& ws, bool initialize_accumulators) {
    constexpr int64_t elem = static_cast<int64_t>(sizeof(T));
    const int64_t output_bytes = plan.accum_map.total_output_elements * elem;

    if (plan.output_permute_active()) {
        // User-facing buffer holds the einsum-order result; the contraction
        // accumulates into an internal-order buffer that becomes workspace.
        if (d_output) {
            plan.d_output_final = d_output;
            plan.owns_output_final = false;
        } else {
            check_cuda(malloc_device_cached(
                           reinterpret_cast<void**>(&plan.d_output_final),
                           output_bytes),
                       "malloc final_output");
            plan.owns_output_final = true;
        }
        if (ws.provided()) {
            plan.d_output = static_cast<T*>(ws.take(output_bytes));
            plan.owns_output = false;
        } else {
            check_cuda(malloc_device_cached(
                           reinterpret_cast<void**>(&plan.d_output), output_bytes),
                       "malloc internal_output");
            plan.owns_output = true;
        }
    } else if (d_output) {
        plan.d_output = d_output;
        plan.owns_output = false;
    } else {
        check_cuda(malloc_device_cached(reinterpret_cast<void**>(&plan.d_output),
                                        output_bytes),
                   "malloc final_output");
        plan.owns_output = true;
    }

    // output_beta != 0 with an active permute is rejected in pipeline_plan,
    // so the internal accumulator is always zeroed when the permute runs.
    if (plan.output_beta == 0 && initialize_accumulators) {
        check_cuda(cudaMemset(plan.d_output, 0, output_bytes), "memset final_output");
    }

    plan.owns_workspace = !ws.provided();

    // Temp buffer
    if (plan.total_pair_output_size > 0) {
        int64_t temp_bytes = plan.total_pair_output_size * elem;
        if (ws.provided()) {
            plan.d_temp_buffer = static_cast<T*>(ws.take(temp_bytes));
        } else {
            check_cuda(malloc_device_cached(
                           reinterpret_cast<void**>(&plan.d_temp_buffer), temp_bytes),
                       "malloc temp_buffer");
        }
        if (initialize_accumulators)
            check_cuda(cudaMemset(plan.d_temp_buffer, 0, temp_bytes),
                       "memset temp_buffer");
    }
}

// --- pipeline_allocate step: patch GPU pointers into all three tiers --------
// Large tier needs d_output + d_temp_buffer bound first; non-streaming large
// work also builds its CUTLASS runners here.
template <typename T>
bool allocate_patch_pointers(PipelinePlan<T>& plan,
                             const DeviceSparseInput<T>& A_dev,
                             const DeviceSparseInput<T>& B_dev) {
    if (!plan.work.large.empty() && !plan.streaming.enabled) {
        // Bind pointers directly into the pre-classified queues, build runners.
        // (Streaming queues are bound in allocate_setup_streaming.)
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            auto& qwork = plan.nonstream_classified.queues[qi];
            if (qwork.empty()) continue;
            bind_queue_pointers(qwork, plan.pairs_host,
                A_dev.d_values, B_dev.d_values, plan.d_temp_buffer,
                plan.d_output, plan.accum_map.h_dest_offsets,
                plan.reduce_plan.h_direct_write);
            plan.nonstream_runners[qi] = prepare_large_gemm_dispatch(
                qwork, T(1), T(1), plan.nonstream_kernels[qi], plan.stream_large);
        }
    }

    if (!plan.work.small.empty())
        bind_small_mems(plan, A_dev, B_dev);

    if (!plan.work.medium.empty()) {
        if (!bind_medium_mems(plan, A_dev, B_dev))
            return pipeline_error("pipeline_allocate: bind_medium_mems failed");
    }
    return true;
}

// Precompute the first ≤2 streaming batches: fill CUTLASS args (+ transpose
// metadata) during setup so execute is pure kernel launches. Cases with >2
// batches fall through to the runtime double-buffer path.
// assemble_meta=true assembles + uploads hypertile metadata into precomp_bufs
// (first bind). pipeline_rebind passes false: the metadata is offset-based
// (base pointers are supplied at kernel launch) and stays valid — only the
// CUTLASS args hold absolute pointers and need re-staging.
template <typename T>
void fill_precomputed_streaming_batches(PipelinePlan<T>& plan,
                                        const GPUSparseInput<T>& A_gpu,
                                        const GPUSparseInput<T>& B_gpu,
                                        const DeviceSparseInput<T>& A_dev,
                                        const DeviceSparseInput<T>& B_dev,
                                        bool assemble_meta) {
    int total_streaming_batches = 0;
    for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi)
        for (const auto& b : plan.streaming.queue_partitions[qi].batches)
            if (!b.problem_indices.empty()) ++total_streaming_batches;

    if (total_streaming_batches > 2) return;

    int gslot = 0;
    for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
        const auto& partition = plan.streaming.queue_partitions[qi];
        const auto& qlarge = plan.streaming.queue_work[qi];
        KernelChoice kernel = plan.streaming.queue_kernels[qi];
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

            if (assemble_meta) {
                TransposeFlatMeta meta_A, meta_B;
                if (plan.layout_plan.plan_A.transpose) {
                    meta_A = assemble_hypertile_batch(
                        batch.unique_blocks_A,
                        A_gpu.block_offsets.data(), A_gpu.block_volumes.data(),
                        plan.shape_map_A,
                        plan.hypertile_index_A, plan.trivial_shapes_A,
                        batch.pack_offset_A,
                        &plan.streaming.precomp_bufs[slot][0], 0);
                }
                if (plan.layout_plan.plan_B.transpose) {
                    meta_B = assemble_hypertile_batch(
                        batch.unique_blocks_B,
                        B_gpu.block_offsets.data(), B_gpu.block_volumes.data(),
                        plan.shape_map_B,
                        plan.hypertile_index_B, plan.trivial_shapes_B,
                        batch.pack_offset_B,
                        &plan.streaming.precomp_bufs[slot][1], 0);
                }
                plan.streaming.precomp_meta[slot][0] = meta_A;
                plan.streaming.precomp_meta[slot][1] = meta_B;
            }

            auto batch_work = build_streaming_batch_work(
                plan, qlarge, batch, expanded_start,
                plan.streaming.precomp_meta[slot][0],
                plan.streaming.precomp_meta[slot][1], d_buf,
                A_dev.d_values, B_dev.d_values);

            fill_large_args_dispatch(
                plan.streaming.args[slot], batch_work,
                T(1), T(1), kernel, 0);

            ++gslot;
        }
    }
    plan.streaming.num_precomp = gslot;
    plan.streaming.precomputed = true;
}

// --- pipeline_allocate step: streaming buffers, streams, precompute ---------
// Moves staged classification into the streaming queues, allocates the
// double-buffered transpose workspace + CUTLASS arg buffers, and precomputes
// the first ≤2 batches so execute is pure kernel launches.
template <typename T>
void allocate_setup_streaming(PipelinePlan<T>& plan,
                              const GPUSparseInput<T>& A_gpu,
                              const GPUSparseInput<T>& B_gpu,
                              const DeviceSparseInput<T>& A_dev,
                              const DeviceSparseInput<T>& B_dev,
                              WorkspaceBump& ws) {
    constexpr int64_t elem = static_cast<int64_t>(sizeof(T));

    // Bind pointer arrays per queue, move staging → streaming queue_work
    for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
        auto& q = plan.staging_classified.queues[qi];
        if (!q.empty())
            bind_queue_pointers(q, plan.pairs_host,
                A_dev.d_values, B_dev.d_values, plan.d_temp_buffer,
                plan.d_output, plan.accum_map.h_dest_offsets,
                plan.reduce_plan.h_direct_write);
        plan.streaming.queue_work[qi] = std::move(q);
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
            if (ws.provided()) {
                plan.streaming.d_buf[b] = static_cast<T*>(ws.take(buf_bytes));
            } else {
                check_cuda(malloc_device_cached(
                               reinterpret_cast<void**>(&plan.streaming.d_buf[b]),
                               buf_bytes),
                           "malloc streaming transpose buffer");
            }
        }
    }

    // Streaming streams + events from the shared context
    {
        auto& sctx = get_stream_context();
        for (int i = 0; i < 2; ++i) {
            plan.streaming.streams[i] = sctx.streaming_streams[i];
            plan.streaming.gemm_done[i] = sctx.gemm_done[i];
            plan.streaming.copy_done[i] = sctx.copy_done[i];
        }
    }

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

    fill_precomputed_streaming_batches(plan, A_gpu, B_gpu, A_dev, B_dev,
                                       /*assemble_meta=*/true);
}

// --- pipeline_allocate step: non-streaming transpose workspace --------------
// Assembles the full-tensor transpose batch and allocates a workspace per
// operand that needs it (hypertile_index + trivial_shapes already built in
// pipeline_prepare).
template <typename T>
void allocate_setup_transpose(PipelinePlan<T>& plan,
                              const GPUSparseInput<T>& A_gpu,
                              const GPUSparseInput<T>& B_gpu,
                              const DeviceSparseInput<T>& A_dev,
                              const DeviceSparseInput<T>& B_dev,
                              WorkspaceBump& ws) {
    constexpr int64_t elem = static_cast<int64_t>(sizeof(T));

    auto build_transpose = [&](const GPUSparseInput<T>& gpu, const auto& op_plan,
                               const BlockShapeMap& smap,
                               const HypertileIndex& idx,
                               const std::vector<uint8_t>& trivial_shapes,
                               TransposeFlatMeta& precomp, T*& d_workspace,
                               int64_t total_elements) {
        if (!op_plan.transpose) return;
        std::vector<int> all_blocks(gpu.num_blocks);
        std::iota(all_blocks.begin(), all_blocks.end(), 0);
        precomp = assemble_hypertile_batch(
            all_blocks, gpu.block_offsets.data(), gpu.block_volumes.data(),
            smap, idx, trivial_shapes,
            {}, nullptr, plan.stream_large);
        if (precomp.kind != TransposeDispatchKind::None) {
            int64_t ws_bytes = total_elements * elem;
            if (ws.provided()) {
                d_workspace = static_cast<T*>(ws.take(ws_bytes));
            } else {
                check_cuda(malloc_device_cached(
                               reinterpret_cast<void**>(&d_workspace), ws_bytes),
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

// --- pipeline_allocate step: output permute metadata -------------------------
// Assembles the full-output hypertile batch once; execute only launches.
// Source and destination use the same block offsets (volumes are
// permutation-invariant), so no pack_offset is needed.
template <typename T>
void allocate_setup_output_permute(PipelinePlan<T>& plan) {
    if (!plan.output_permute_active()) return;
    const auto& am = plan.accum_map;
    std::vector<int> all_blocks(static_cast<size_t>(am.num_output_blocks));
    std::iota(all_blocks.begin(), all_blocks.end(), 0);
    plan.output_permute_meta = assemble_hypertile_batch(
        all_blocks, am.h_output_block_offsets, am.h_output_block_volumes,
        plan.shape_map_out, plan.hypertile_index_out, plan.trivial_shapes_out,
        {}, nullptr, plan.stream_large);
}

// ============================================================================
// Phase 3: Bind workspace buffers, patch GPU pointers, upload to device.
// If d_output/workspace are null, self-allocates (backward-compat behavior).
//
// `initialize_accumulators` must be true for any plan that will be executed:
// every GEMM tier accumulates with beta=1, so an unzeroed d_output /
// d_temp_buffer produces garbage.  Pass false ONLY to isolate the host
// planning cost from the device zero-fill in benchmarks (which then perform
// the equivalent memsets themselves); the 8-argument overload below defaults
// it to true.
// ============================================================================
template <typename T>
bool pipeline_allocate(
    PipelinePlan<T>& plan,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    T* d_output,
    void* workspace,
    int64_t workspace_bytes,
    bool initialize_accumulators)
{
    if (plan.num_pairs == 0) return true;
    if (!plan.prepare_done)
        return pipeline_error("pipeline_allocate: pipeline_prepare not called");

    // Snapshot free memory before allocations for accurate peak measurement
    size_t free_before = 0, total_ignore = 0;
    cudaMemGetInfo(&free_before, &total_ignore);

    using alloc_clock = std::chrono::high_resolution_clock;

    // --- GPU uploads deferred from pipeline_plan ---
    double bind_upload = 0, bind_patch = 0, bind_stream = 0;
    auto tb0 = alloc_clock::now();
    upload_reduction_plan(plan.reduce_plan);
    bind_upload = detail_timing::ms_since(tb0);

    {
        auto& sctx = get_stream_context();
        plan.stream_large = sctx.stream_large;
        plan.stream_small = sctx.stream_small;
        plan.stream_medium = sctx.stream_medium;
        plan.transpose_done_flat = sctx.transpose_done_flat;
        plan.transpose_done_tiled = sctx.transpose_done_tiled;
    }

    WorkspaceBump ws{workspace, workspace_bytes};

    // The default preserves the API's fresh-execute semantics.  The benchmark
    // disables initialization here, then zeros the exact same accumulator
    // regions after the host-preparation endpoint so the two costs are visible.
    allocate_bind_output(plan, d_output, ws, initialize_accumulators);
    allocate_setup_output_permute(plan);

    // Init transposed values
    plan.tv.d_A = A_dev.d_values;
    plan.tv.d_B = B_dev.d_values;
    plan.tv.d_workspace_A = nullptr;
    plan.tv.d_workspace_B = nullptr;
    plan.tv.gemm_mode = GemmMode::NN;

    // --- Patch GPU pointers into all tiers (needs d_output + d_temp_buffer) ---
    auto tb1 = alloc_clock::now();
    if (!allocate_patch_pointers(plan, A_dev, B_dev))
        return false;
    bind_patch = detail_timing::ms_since(tb1);

    // --- Streaming / transpose workspace setup ---
    auto tb2 = alloc_clock::now();
    if (plan.streaming.enabled) {
        allocate_setup_streaming(plan, A_gpu, B_gpu, A_dev, B_dev, ws);
    } else if (plan.needs_transpose) {
        allocate_setup_transpose(plan, A_gpu, B_gpu, A_dev, B_dev, ws);
    }
    bind_stream = detail_timing::ms_since(tb2);

    // Measure actual pipeline allocation via cudaMemGetInfo delta
    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total_ignore);
    plan.measured_alloc_bytes = static_cast<int64_t>(free_before - free_after);

    if (plan.timing) {
        plan.timing->bind_upload_ms = bind_upload;
        plan.timing->bind_patch_ms = bind_patch;
        plan.timing->bind_stream_ms = bind_stream;
        plan.timing->bind_ms = bind_upload + bind_patch + bind_stream;
    }

    cudaDeviceSynchronize();
    plan.allocate_done = true;
    return true;
}

// Source- and ABI-compatible public entry point.  Normal callers always get
// fresh accumulator initialization; the overload above exists so timing tools
// can expose that device work separately without changing execution semantics.
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
    return pipeline_allocate(plan, A_gpu, B_gpu, A_dev, B_dev,
                             d_output, workspace, workspace_bytes,
                             /*initialize_accumulators=*/true);
}

// ============================================================================
// Re-zero the accumulator for a repeated pipeline_execute with unchanged
// pointers (output_beta == 0 semantics). The temp buffer needs no reset — the
// reduce kernel restores its slots to zero after reading them.
// ============================================================================
template <typename T>
void pipeline_reset_output(PipelinePlan<T>& plan) {
    if (plan.num_pairs == 0 || plan.output_beta != 0) return;
    const int64_t output_bytes =
        plan.accum_map.total_output_elements * static_cast<int64_t>(sizeof(T));
    check_cuda(cudaMemset(plan.d_output, 0, output_bytes),
               "pipeline_reset_output memset");
}

// ============================================================================
// Rebind an allocated plan to new device base pointers (same metadata).
// Re-runs only the pointer-staging work: everything metadata-derived (pair
// matching, tiers, accum map, hypertile indexes, reduction plan, streams,
// uploaded shapes) is reused from the first pipeline_allocate. Buffers the
// plan self-allocated on the first bind (null d_output/workspace) are kept;
// caller-provided ones are re-carved from the new workspace in the same
// order as pipeline_allocate, so offsets match the original layout.
// Also re-zeroes the accumulator (output_beta == 0), so a rebind is always
// followed by a correct fresh execute.
// ============================================================================
template <typename T>
bool pipeline_rebind(
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
    if (!plan.allocate_done)
        return pipeline_error("pipeline_rebind: pipeline_allocate not called");

    constexpr int64_t elem = static_cast<int64_t>(sizeof(T));
    const int64_t output_bytes = plan.accum_map.total_output_elements * elem;
    WorkspaceBump ws{workspace, workspace_bytes};

    // --- Output buffers (same carve order as allocate_bind_output) ---
    if (plan.output_permute_active()) {
        if (!plan.owns_output_final) {
            if (!d_output)
                return pipeline_error(
                    "pipeline_rebind: plan was bound to a user output buffer");
            plan.d_output_final = d_output;
        }
        if (!plan.owns_output)
            plan.d_output = static_cast<T*>(ws.take(output_bytes));
    } else if (!plan.owns_output) {
        if (!d_output)
            return pipeline_error(
                "pipeline_rebind: plan was bound to a user output buffer");
        plan.d_output = d_output;
    }
    if (plan.output_beta == 0)
        check_cuda(cudaMemset(plan.d_output, 0, output_bytes),
                   "rebind memset output");

    // --- Temp buffer: re-carve when user-provided; must be re-zeroed (the
    // reduce-kernel invariant only holds within one buffer's lifetime) ---
    if (plan.total_pair_output_size > 0) {
        const int64_t temp_bytes = plan.total_pair_output_size * elem;
        if (!plan.owns_workspace)
            plan.d_temp_buffer = static_cast<T*>(ws.take(temp_bytes));
        check_cuda(cudaMemset(plan.d_temp_buffer, 0, temp_bytes),
                   "rebind memset temp");
    }

    // --- Re-stage absolute pointers into all tiers ---
    if (!allocate_patch_pointers(plan, A_dev, B_dev))
        return false;

    if (plan.streaming.enabled) {
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            auto& q = plan.streaming.queue_work[qi];
            if (!q.empty())
                bind_queue_pointers(q, plan.pairs_host,
                    A_dev.d_values, B_dev.d_values, plan.d_temp_buffer,
                    plan.d_output, plan.accum_map.h_dest_offsets,
                    plan.reduce_plan.h_direct_write);
        }
        // Streaming transpose buffers follow owns_workspace (same carve
        // order/count as allocate_setup_streaming).
        if (!plan.owns_workspace && plan.needs_transpose &&
            plan.streaming.buf_cap > 0) {
            const int64_t buf_bytes = plan.streaming.buf_cap * elem;
            const int num_bufs = plan.streaming.d_buf[1] ? 2 : 1;
            for (int b = 0; b < num_bufs; ++b)
                plan.streaming.d_buf[b] = static_cast<T*>(ws.take(buf_bytes));
        }
        // Precomputed fast path bakes absolute pointers into CUTLASS args —
        // re-fill them, reusing the offset-based transpose metadata.
        if (plan.streaming.precomputed)
            fill_precomputed_streaming_batches(plan, A_gpu, B_gpu,
                                               A_dev, B_dev,
                                               /*assemble_meta=*/false);
    } else if (plan.needs_transpose) {
        // Non-streaming transpose workspaces (carved in the same order as
        // allocate_setup_transpose; metadata itself is offset-based).
        if (!plan.owns_workspace) {
            if (plan.tv.d_workspace_A)
                plan.tv.d_workspace_A =
                    static_cast<T*>(ws.take(A_dev.total_elements * elem));
            if (plan.tv.d_workspace_B)
                plan.tv.d_workspace_B =
                    static_cast<T*>(ws.take(B_dev.total_elements * elem));
        }
    }

    // Reset the transposed-values view so stage_transpose/stage_patch_pointers
    // start from the new inputs.
    plan.tv.d_A = A_dev.d_values;
    plan.tv.d_B = B_dev.d_values;

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
    if (!pipeline_prepare(plan, A_gpu, B_gpu))
        return false;
    if (timing) timing->prepare_ms = detail_timing::ms_since(t1);
    auto t2 = clock::now();
    bool ok = pipeline_allocate(plan, A_gpu, B_gpu, A_dev, B_dev);
    if (timing) timing->allocate_ms = detail_timing::ms_since(t2);
    // Diagnostics AFTER the timing endpoints: populate_timing_stats rescans
    // the per-tier problem vectors and block dims, which must not pollute the
    // measured setup phases (CPU setup time is the tuning signal). It needs
    // allocate-time state (measured_alloc_bytes, streaming metadata), so it
    // cannot run earlier.
    if (ok && timing)
        populate_timing_stats(*timing, plan, A_dev, B_dev, A_gpu, B_gpu);
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
        if (plan.total_warps > 0 && plan.small_num_dots > 0)
            timing->small_kernel_kind = "warp+dot";
        else if (plan.total_warps > 0) timing->small_kernel_kind = "warp";
        else if (plan.small_num_dots > 0) timing->small_kernel_kind = "dot";
        else timing->small_kernel_kind = nullptr;
    }

    cudaDeviceSynchronize();
    if (timing) { timing->contraction_ms = detail_timing::ms_since(t_stage); }

    // --- Accumulation ---
    t_stage = clock::now();

    stage_accumulate(plan, 0);
    cudaDeviceSynchronize();

    if (timing) { timing->accumulation_ms = detail_timing::ms_since(t_stage); }

    // --- Output permute: internal order → einsum output order ---
    if (plan.output_permute_active()) {
        t_stage = clock::now();
        constexpr int64_t elem = static_cast<int64_t>(sizeof(T));
        if (plan.output_permute_copy_first) {
            // Trivial-shape blocks are not touched by the permute kernel;
            // cover them with one full-buffer copy (rare mixed case).
            check_cuda(cudaMemcpyAsync(
                           plan.d_output_final, plan.d_output,
                           plan.accum_map.total_output_elements * elem,
                           cudaMemcpyDeviceToDevice, 0),
                       "output permute pre-copy");
        }
        launch_batched_transpose<T>(plan.d_output, plan.d_output_final,
                                    plan.output_permute_meta, 0);
        cudaDeviceSynchronize();
        if (timing) { timing->output_permute_ms = detail_timing::ms_since(t_stage); }
    } else if (timing) {
        timing->output_permute_ms = 0.0;
    }

    if (timing) { timing->total_ms = detail_timing::ms_since(t_total); }
}

// Result contract: d_output holds packed blocks, each block row-major in
// einsum output label order; output_block_coords are in that same order.
// When the einsum order matches internal order (or the permutation moves no
// data), the internal buffer is handed out directly.
template <typename T>
GeneralSparseResult<T> pipeline_extract_result(PipelinePlan<T>& plan) {
    GeneralSparseResult<T> result;
    if (plan.output_permute_active()) {
        result.d_output = plan.d_output_final;
        result.owns_output = plan.owns_output_final;
        plan.d_output_final = nullptr;
    } else {
        result.d_output = plan.d_output;
        result.owns_output = plan.owns_output;
        plan.d_output = nullptr;
    }

    if (plan.num_pairs == 0) {
        result.total_output_elements = 0;
        result.num_output_blocks = 0;
        result.output_rank = plan.output_rank;
        return result;
    }

    const int32_t rank = plan.accum_map.output_rank;
    result.total_output_elements = plan.accum_map.total_output_elements;
    result.num_output_blocks = plan.accum_map.num_output_blocks;
    result.output_rank = rank;
    result.output_offsets.resize(plan.accum_map.num_output_blocks);
    result.output_volumes.resize(plan.accum_map.num_output_blocks);
    result.output_block_coords.resize(plan.accum_map.num_output_blocks * rank);

    const int32_t* i2o = plan.config.internal_to_output;
    for (int64_t i = 0; i < plan.accum_map.num_output_blocks; ++i) {
        result.output_offsets[i] = plan.accum_map.h_output_block_offsets[i];
        result.output_volumes[i] = plan.accum_map.h_output_block_volumes[i];
        for (int32_t d = 0; d < rank; ++d) {
            result.output_block_coords[i * rank + i2o[d]] =
                plan.accum_map.h_output_block_coords[i * rank + d];
        }
    }

    return result;
}

} // namespace bsptc
