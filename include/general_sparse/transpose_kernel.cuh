#pragma once

#include "gpu_sparse_device.cuh"
#include "preprocessing_kernel.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// GPU kernels for transposing all blocks of a sparse tensor in parallel.
// Uses batched carry-over kernels with per-batch multi-CTA parallelism.
namespace bsptc {

// Per-CTA work assignment. Multiple CTAs can share a batch; each strides
// through the batch's elements using (cta_local, ctas_in_batch).
struct CTAWork {
    int desc_start;     // first desc index in this batch
    int desc_count;     // number of descs in this batch
    int cta_local;      // this CTA's index within the batch (0..ctas_in_batch-1)
    int ctas_in_batch;  // total CTAs assigned to this batch
};

// Per-block descriptor for batched generic-permute transpose.
// Packed layout: fixed fields + variable-length head_dims[head_rank] + dst_stride[head_rank].
// stride_per_desc = 4 + 2*head_rank (int64_t units).
struct GenericPermuteTransposeDesc {
    int64_t src_data_offset; // offset into src values array
    int64_t dst_data_offset; // offset into dst values array
    int64_t tail_volume;     // product of unmoved tail dimensions
    int64_t work_items;      // total elements in this block (= head_volume * tail_volume)
    // Followed by: int64_t head_dims[head_rank]   (cast to int32 for decomposition)
    //              int64_t dst_stride[head_rank]   (precomputed dst_stride_for_src)
};

// Batched generic-permute transpose kernel with per-batch multi-CTA parallelism.
template <typename T, int BLOCK_SIZE = 256>
__global__ void transpose_blocks_generic_permute_batched_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int64_t* __restrict__ descs,    // packed descriptors
    const CTAWork* __restrict__ cta_work,
    dim_t head_rank,
    dim_t stride_per_desc)                // = 3 + 2*head_rank
{
    const CTAWork w = cta_work[blockIdx.x];

    int elem = w.cta_local * BLOCK_SIZE + threadIdx.x;
    const int stride = BLOCK_SIZE * w.ctas_in_batch;
    for (int i = 0; i < w.desc_count; ++i) {
        const int64_t* desc = descs + static_cast<int64_t>(w.desc_start + i) * stride_per_desc;
        const int64_t src_data_offset = desc[0];
        const int64_t dst_data_offset = desc[1];
        const int64_t tail_volume = desc[2];
        const int work = static_cast<int>(desc[3]);
        const int64_t* head_dims = desc + 4;
        const int64_t* dst_stride = desc + 4 + head_rank;

        while (elem < work) {
            // Decompose: elem → (head_idx, tail_idx)
            int head_idx = elem / static_cast<int>(tail_volume);
            int tail_idx = elem - head_idx * static_cast<int>(tail_volume);

            // Permute head index using precomputed dst_stride_for_src
            int64_t dst_head_idx = 0;
            int32_t rem = head_idx;
            for (int32_t d = head_rank - 1; d >= 0; --d) {
                int32_t dim = static_cast<int32_t>(head_dims[d]);
                int32_t c = rem % dim;
                rem /= dim;
                dst_head_idx += static_cast<int64_t>(c) * dst_stride[d];
            }

            int64_t src_linear = static_cast<int64_t>(head_idx) * tail_volume + tail_idx;
            int64_t dst_linear = dst_head_idx * tail_volume + tail_idx;

            dst[dst_data_offset + dst_linear] = src[src_data_offset + src_linear];

            elem += stride;
        }
        elem -= work;
    }
}


// ND flat-collapse SMEM generic permute kernel.
// Collapses multiple contiguous source dims into flat_I and multiple contiguous
// dest dims into flat_O. N-dimensional decomposition happens only in the ty
// (slow/warp-constant) direction, so it's essentially free.
constexpr int ND_MAX_COLLAPSE = 3;

struct NDPermuteTileWork {
    int64_t src_base;                       // data_offset + batch source offset
    int64_t dst_base;                       // data_offset + batch dest offset
    int32_t tile_i;                         // tile start in flat_I space
    int32_t tile_o;                         // tile start in flat_O space
    int32_t flat_I_size;                    // total flat I extent
    int32_t flat_O_size;                    // total flat O extent
    int32_t O_shape[ND_MAX_COLLAPSE];       // O dim sizes (innermost-first)
    int64_t O_src_stride[ND_MAX_COLLAPSE];  // source strides of O dims
    int32_t I_shape[ND_MAX_COLLAPSE];       // I dim sizes (innermost-first)
    int64_t I_dst_stride[ND_MAX_COLLAPSE];  // dest strides of I dims
    // Unused dims: shape=1, stride=0 (no-op in unrolled decomposition)
};

// SMEM-tiled ND generic permute kernel.
// Always unrolls ND_MAX_COLLAPSE=3 iterations — unused dims have shape=1, stride=0.
// Phase 1 (read): tx sweeps flat_I (coalesced source reads).
//   Decompose global_O = tile_o + row into O coords, compute source offset.
// Phase 2 (write): tx sweeps flat_O (coalesced dest writes).
//   Decompose global_I = tile_i + row into I coords, compute dest offset.
// Each CTA strides through multiple tiles (gridDim.x may be < total_tiles).
template <typename T, int TILE = 32, int BLOCK_SIZE = 256>
__global__ void generic_permute_nd_tile_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const NDPermuteTileWork* __restrict__ tile_work,
    int total_tiles)
{
    constexpr int ROWS_PER_ITER = BLOCK_SIZE / TILE;  // 8 for 256/32
    __shared__ T smem[TILE][TILE + 1];

    const int tx = threadIdx.x % TILE;
    const int ty = threadIdx.x / TILE;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const NDPermuteTileWork& tw = tile_work[tile_idx];
        const int ti = tw.tile_i, to = tw.tile_o;
        const int tile_cols_I = min(TILE, tw.flat_I_size - ti);
        const int tile_rows_O = min(TILE, tw.flat_O_size - to);

        // Phase 1: coalesced read (tx along flat_I, contiguous in source)
        // For each row (O index), decompose into O coords → source offset
        for (int row = ty; row < tile_rows_O; row += ROWS_PER_ITER) {
            if (tx < tile_cols_I) {
                int32_t global_o = to + row;
                int64_t O_offset = 0;
                #pragma unroll
                for (int d = 0; d < ND_MAX_COLLAPSE; ++d) {
                    int32_t coord = global_o % tw.O_shape[d];
                    global_o /= tw.O_shape[d];
                    O_offset += static_cast<int64_t>(coord) * tw.O_src_stride[d];
                }
                int64_t in_idx = tw.src_base + O_offset + (ti + tx);
                smem[row][tx] = src[in_idx];
            }
        }

        __syncthreads();

        // Phase 2: transposed SMEM read, coalesced write (tx along flat_O, contiguous in dest)
        // For each row (I index), decompose into I coords → dest offset
        for (int row = ty; row < tile_cols_I; row += ROWS_PER_ITER) {
            if (tx < tile_rows_O) {
                T val = smem[tx][row];
                int32_t global_i = ti + row;
                int64_t I_offset = 0;
                #pragma unroll
                for (int d = 0; d < ND_MAX_COLLAPSE; ++d) {
                    int32_t coord = global_i % tw.I_shape[d];
                    global_i /= tw.I_shape[d];
                    I_offset += static_cast<int64_t>(coord) * tw.I_dst_stride[d];
                }
                int64_t out_idx = tw.dst_base + I_offset + (to + tx);
                dst[out_idx] = val;
            }
        }

        __syncthreads();  // ensure stores complete before next tile overwrites SMEM
    }
}

// Build CTAWork array from batch ranges with per-batch multi-CTA parallelism.
// Each batch gets max(1, ceil(batch_elems / elems_per_cta)) CTAs.
inline std::vector<CTAWork> build_cta_work(
    const int64_t* work_per_item, int num_items,
    int target_work, int /*block_size*/)
{
    // First pack descs into batches (same greedy packing as before)
    std::vector<int2> ranges;
    if (num_items > 0) {
        int i = 0;
        while (i < num_items) {
            int start = i;
            int64_t total = 0;
            int count = 0;
            while (i < num_items && (count == 0 || total + work_per_item[i] <= target_work)) {
                total += work_per_item[i];
                ++count;
                ++i;
            }
            ranges.push_back(make_int2(start, count));
        }
    }

    // Compute per-batch element totals and assign CTAs
    constexpr int ELEMS_PER_CTA = 256 * 32;  // 8192 elements per CTA
    std::vector<CTAWork> cta_work;
    for (const auto& r : ranges) {
        int64_t batch_elems = 0;
        for (int j = 0; j < r.y; ++j)
            batch_elems += work_per_item[r.x + j];
        int num_ctas = static_cast<int>((batch_elems + ELEMS_PER_CTA - 1) / ELEMS_PER_CTA);
        if (num_ctas < 1) num_ctas = 1;
        for (int c = 0; c < num_ctas; ++c)
            cta_work.push_back({r.x, r.y, c, num_ctas});
    }
    return cta_work;
}

} // namespace bsptc
