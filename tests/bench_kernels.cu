// Synthetic kernel benchmarks for transpose and accumulation kernels.
// No tensor files or cuTENSOR needed — generates block metadata on host,
// allocates device buffers, and times individual kernels with cudaEvent.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "general_sparse/transpose_kernel.cuh"
#include "general_sparse/gpu_accumulator.cuh"
#include "general_sparse/gpu_sparse_device.cuh"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/gemm_layout_check.hpp"
#include "general_sparse/kernels/transpose/nd_hypertile.cuh"
#include "general_sparse/kernels/transpose/hypertile_index_builder.hpp"
#include "general_sparse/block_shape_map.hpp"

using namespace bsptc;

// ---------------------------------------------------------------------------
// Helper: build hypertile GPU data from HypertileIndex for standalone benchmarks
// ---------------------------------------------------------------------------

struct HypertileGPUSlice {
    void* d_shapes;
    void* d_mems;
    int32_t* d_prefix_sum;
    int rank, num_tasks, total_tiles;
};

struct HypertileGPUData {
    std::vector<HypertileGPUSlice> slices;
    void free_gpu() {
        for (auto& s : slices) {
            cudaFree(s.d_shapes); cudaFree(s.d_mems); cudaFree(s.d_prefix_sum);
        }
        slices.clear();
    }
};

template <typename T>
HypertileGPUData build_hypertile_gpu(
    const GPUSparseInput<T>& gpu, const HypertileIndex& index,
    const BlockShapeMap& shape_map, cudaStream_t stream)
{
    HypertileGPUData result;
    if (!index.valid || index.num_slices == 0) return result;

    // Accumulate mems per slice
    struct SliceAccum {
        std::vector<HypertileMem> mems;
        int32_t total_tiles = 0;
    };
    std::vector<SliceAccum> accums(index.num_slices);

    for (int b = 0; b < static_cast<int>(gpu.num_blocks); ++b) {
        if (gpu.block_volumes[b] == 0) continue;
        uint32_t shape_id = shape_map.block_to_shape[b];
        const auto& entry = index.per_shape[shape_id];
        if (entry.trivial) continue;

        HypertileMem mem{};
        mem.src_base = gpu.block_offsets[b];
        mem.dst_base = gpu.block_offsets[b];
        mem.shape_idx = entry.shape_idx;

        auto& acc = accums[entry.slice_idx];
        acc.mems.push_back(mem);
        acc.total_tiles += entry.num_tiles;
    }

    // Sort by shape_idx and build prefix_sum, then upload
    for (int s = 0; s < index.num_slices; ++s) {
        auto& acc = accums[s];
        if (acc.mems.empty()) continue;

        // Counting sort by shape_idx
        if (acc.mems.size() > 1) {
            int max_shape = index.slices[s].num_shapes;
            std::vector<int> counts(max_shape, 0);
            for (const auto& m : acc.mems) counts[m.shape_idx]++;
            std::vector<int> offsets(max_shape, 0);
            for (int i = 1; i < max_shape; ++i) offsets[i] = offsets[i-1] + counts[i-1];
            std::vector<HypertileMem> sorted(acc.mems.size());
            for (const auto& m : acc.mems) sorted[offsets[m.shape_idx]++] = m;
            acc.mems = std::move(sorted);
        }

        // Build prefix_sum
        const auto& si = index.slices[s];
        size_t shape_stride = si.shape_bytes / si.num_shapes;
        std::vector<int32_t> prefix_sum;
        prefix_sum.push_back(0);
        for (const auto& m : acc.mems) {
            uint32_t num_tiles;
            std::memcpy(&num_tiles,
                        si.h_shape_bytes.data() + m.shape_idx * shape_stride,
                        sizeof(uint32_t));
            prefix_sum.push_back(prefix_sum.back() + num_tiles);
        }

        HypertileGPUSlice gs;
        gs.rank = si.rank;
        gs.num_tasks = static_cast<int>(acc.mems.size());
        gs.total_tiles = acc.total_tiles;

        size_t sb = si.shape_bytes;
        size_t mb = acc.mems.size() * sizeof(HypertileMem);
        size_t ps = prefix_sum.size() * sizeof(int32_t);
        cudaMalloc(&gs.d_shapes, sb);
        cudaMemcpyAsync(gs.d_shapes, si.h_shape_bytes.data(), sb,
                        cudaMemcpyHostToDevice, stream);
        cudaMalloc(&gs.d_mems, mb);
        cudaMemcpyAsync(gs.d_mems, acc.mems.data(), mb,
                        cudaMemcpyHostToDevice, stream);
        cudaMalloc(&gs.d_prefix_sum, ps);
        cudaMemcpyAsync(gs.d_prefix_sum, prefix_sum.data(), ps,
                        cudaMemcpyHostToDevice, stream);
        result.slices.push_back(gs);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

struct BenchResult { float median_ms, min_ms, max_ms; };

template <typename Fn>
BenchResult gpu_bench(Fn fn, int warmup, int repeat, cudaStream_t stream) {
    for (int i = 0; i < warmup; i++) { fn(); }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(repeat);
    for (int i = 0; i < repeat; i++) {
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(times.begin(), times.end());
    return {times[repeat / 2], times[0], times[repeat - 1]};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Factor volume into rank roughly-equal dims.
// First rank-1 dims are round(volume^(1/rank)), last dim absorbs remainder.
static std::vector<int64_t> factor_dims(int64_t volume, int rank) {
    std::vector<int64_t> dims(rank);
    double base = std::pow((double)volume, 1.0 / rank);
    int64_t prod = 1;
    for (int i = 0; i < rank - 1; i++) {
        dims[i] = std::max((int64_t)2, (int64_t)std::round(base));
        prod *= dims[i];
    }
    dims[rank - 1] = std::max((int64_t)1, volume / prod);
    return dims;
}

static int64_t product(const std::vector<int64_t>& dims) {
    int64_t p = 1;
    for (auto d : dims) p *= d;
    return p;
}

// Generate per-block volumes for uniform or skewed distribution.
// Skewed: 20% get 4x, 30% get 1x, 50% get 1/4x (min mn_threshold).
static std::vector<int64_t> generate_volumes(int num_blocks, int64_t nominal, bool skewed,
                                              int64_t min_vol = 4096) {
    std::vector<int64_t> vols(num_blocks);
    if (!skewed) {
        std::fill(vols.begin(), vols.end(), nominal);
    } else {
        for (int i = 0; i < num_blocks; i++) {
            float frac = static_cast<float>(i) / static_cast<float>(num_blocks);
            if (frac < 0.2f)
                vols[i] = nominal * 4;
            else if (frac < 0.5f)
                vols[i] = nominal;
            else
                vols[i] = std::max(min_vol, nominal / 4);
        }
    }
    return vols;
}

// ---------------------------------------------------------------------------
// Transpose benchmark
// ---------------------------------------------------------------------------

template <typename T>
static void run_transpose_bench(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Transpose Kernel Benchmark (Generic Permute) ===\n");
    printf("%-14s %6s %7s %4s %-8s %10s %8s\n",
           "Kind", "Blocks", "Volume", "Rank", "Dist", "Median(ms)", "GB/s");
    printf("%-14s %6s %7s %4s %-8s %10s %8s\n",
           "----", "------", "------", "----", "----", "----------", "----");

    int num_blocks_arr[] = {100, 1000, 10000};
    int64_t volume_arr[] = {4096, 16384, 65536};
    int rank_arr[] = {3, 4, 6};
    const char* dists[] = {"uniform", "skewed"};

    for (auto num_blocks : num_blocks_arr) {
        for (auto nominal_vol : volume_arr) {
            for (auto rank : rank_arr) {
                for (int di = 0; di < 2; di++) {
                    bool skewed = (di == 1);

                    auto volumes = generate_volumes(num_blocks, nominal_vol, skewed);

                    // Build offsets, dims
                    std::vector<int64_t> h_offsets(num_blocks);
                    std::vector<int64_t> h_dims(num_blocks * rank);
                    int64_t total = 0;
                    for (int b = 0; b < num_blocks; b++) {
                        h_offsets[b] = total;
                        auto dims = factor_dims(volumes[b], rank);
                        for (int r = 0; r < rank; r++)
                            h_dims[b * rank + r] = dims[r];
                        total += product(dims);
                    }

                    // Build permutation (generic permute: reverse head dims)
                    std::vector<int32_t> h_perm(rank);
                    int unmoved_tail = std::min(2, rank - 1);
                    for (int i = 0; i < rank - unmoved_tail; i++)
                        h_perm[i] = rank - unmoved_tail - 1 - i;
                    for (int i = rank - unmoved_tail; i < rank; i++)
                        h_perm[i] = i;

                    // Allocate device memory
                    T *d_src, *d_dst;

                    cudaError_t err = cudaMalloc(&d_src, total * sizeof(T));
                    if (err == cudaErrorMemoryAllocation) {
                        printf("%-14s %6d %7ld %4d %-8s %10s %8s\n",
                               "GP_Batched", num_blocks, nominal_vol, rank, dists[di],
                               "OOM", "-");
                        cudaGetLastError(); // clear
                        continue;
                    }
                    check_cuda(err, "malloc d_src");
                    err = cudaMalloc(&d_dst, total * sizeof(T));
                    if (err == cudaErrorMemoryAllocation) {
                        cudaFree(d_src); cudaGetLastError();
                        printf("%-14s %6d %7ld %4d %-8s %10s %8s\n",
                               "GP_Batched", num_blocks, nominal_vol, rank, dists[di],
                               "OOM", "-");
                        continue;
                    }
                    check_cuda(err, "malloc d_dst");

                    // Fill src with 1s (content irrelevant)
                    cudaMemsetAsync(d_src, 0x3f, total * sizeof(T), stream); // ~1.0f-ish
                    cudaStreamSynchronize(stream);

                    // Build batched precomp for GenericPermuteBatched
                    int kl_head_rank = rank - unmoved_tail;
                    int kl_stride = 4 + 2 * kl_head_rank;
                    std::vector<int32_t> inv_perm(kl_head_rank);
                    for (int sd = 0; sd < kl_head_rank; ++sd)
                        inv_perm[h_perm[sd]] = sd;

                    std::vector<int64_t> kl_desc_data(num_blocks * kl_stride);
                    std::vector<int64_t> work_per(num_blocks);
                    for (int b = 0; b < num_blocks; b++) {
                        const int64_t* dims = &h_dims[b * rank];
                        int64_t tail_vol = 1;
                        for (int r = kl_head_rank; r < rank; r++) tail_vol *= dims[r];

                        int64_t dst_strides[16];
                        dst_strides[kl_head_rank - 1] = 1;
                        for (int dd = kl_head_rank - 2; dd >= 0; --dd)
                            dst_strides[dd] = dst_strides[dd+1] * dims[inv_perm[dd+1]];

                        int64_t* desc = &kl_desc_data[b * kl_stride];
                        desc[0] = h_offsets[b];
                        desc[1] = h_offsets[b];  // dst_data_offset = src (identity)
                        desc[2] = tail_vol;
                        desc[3] = volumes[b];
                        for (int d = 0; d < kl_head_rank; d++)
                            desc[4 + d] = dims[d];
                        for (int sd = 0; sd < kl_head_rank; sd++)
                            desc[4 + kl_head_rank + sd] = dst_strides[h_perm[sd]];
                        work_per[b] = volumes[b];
                    }

                    auto h_cta = build_cta_work(
                        work_per.data(), num_blocks, 256 * 64, 256);
                    int kl_num_ctas = (int)h_cta.size();

                    int64_t *d_kl_descs;
                    CTAWork *d_kl_cta;
                    cudaMalloc(&d_kl_descs, kl_desc_data.size() * sizeof(int64_t));
                    cudaMemcpyAsync(d_kl_descs, kl_desc_data.data(),
                        kl_desc_data.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
                    cudaMalloc(&d_kl_cta, h_cta.size() * sizeof(CTAWork));
                    cudaMemcpyAsync(d_kl_cta, h_cta.data(),
                        h_cta.size() * sizeof(CTAWork),
                        cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);

                    auto result = gpu_bench([&]() {
                        transpose_blocks_generic_permute_batched_kernel<T, 256>
                            <<<kl_num_ctas, 256, 0, stream>>>(
                                d_src, d_dst, d_kl_descs, d_kl_cta,
                                kl_head_rank, kl_stride);
                    }, warmup, repeat, stream);

                    // Throughput: read + write
                    double bytes = (double)total * sizeof(T) * 2.0;
                    double gbps = bytes / (result.median_ms * 1e6);

                    printf("%-14s %6d %7ld %4d %-8s %10.3f %8.1f\n",
                           "GP_Batched", num_blocks, nominal_vol, rank, dists[di],
                           result.median_ms, gbps);

                    cudaFree(d_kl_descs);
                    cudaFree(d_kl_cta);
                    cudaFree(d_src);
                    cudaFree(d_dst);
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Accumulation benchmark
// ---------------------------------------------------------------------------

template <typename T>
static void run_accum_bench(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Accumulation Kernel Benchmark ===\n");
    printf("%-14s %7s %9s %7s %-8s %10s %8s\n",
           "Kernel", "OutBlks", "Pairs/Blk", "Volume", "Dist", "Median(ms)", "GB/s");
    printf("%-14s %7s %9s %7s %-8s %10s %8s\n",
           "------", "-------", "---------", "------", "----", "----------", "----");

    int out_blocks_arr[] = {10, 100, 1000};
    int ppb_arr[] = {2, 4, 8};
    int64_t volume_arr[] = {4096, 16384, 65536};
    const char* kernel_names[] = {"noatomic"};
    const char* dists[] = {"uniform", "skewed"};

    for (int ki = 0; ki < 1; ki++) {
        for (auto num_out : out_blocks_arr) {
            for (auto ppb : ppb_arr) {
                for (auto nominal_vol : volume_arr) {
                    for (int di = 0; di < 2; di++) {
                        bool skewed = (di == 1);

                        // Per-block output volumes and pair counts
                        std::vector<int64_t> out_vols(num_out);
                        std::vector<int> block_ppb(num_out);
                        if (!skewed) {
                            std::fill(out_vols.begin(), out_vols.end(), nominal_vol);
                            std::fill(block_ppb.begin(), block_ppb.end(), ppb);
                        } else {
                            for (int b = 0; b < num_out; b++) {
                                float frac = static_cast<float>(b) / static_cast<float>(num_out);
                                if (frac < 0.2f) {
                                    out_vols[b] = nominal_vol * 4;
                                    block_ppb[b] = ppb * 2;
                                } else if (frac < 0.5f) {
                                    out_vols[b] = nominal_vol;
                                    block_ppb[b] = ppb;
                                } else {
                                    out_vols[b] = std::max((int64_t)4096, nominal_vol / 4);
                                    block_ppb[b] = std::max(1, ppb / 2);
                                }
                            }
                        }

                        // Compute total pairs and output size
                        int64_t total_pairs = 0;
                        for (int b = 0; b < num_out; b++) total_pairs += block_ppb[b];
                        int64_t total_out = 0;
                        for (int b = 0; b < num_out; b++) total_out += out_vols[b];

                        // Build dest_offsets for output blocks (prefix sum)
                        std::vector<int64_t> h_out_offsets(num_out);
                        {
                            int64_t off = 0;
                            for (int b = 0; b < num_out; b++) {
                                h_out_offsets[b] = off;
                                off += out_vols[b];
                            }
                        }

                        // Build per-pair data: temp offsets, BlockPairInfo
                        std::vector<BlockPairInfo> h_pairs(total_pairs);
                        {
                            int64_t p = 0, temp_off = 0;
                            for (int b = 0; b < num_out; b++) {
                                for (int j = 0; j < block_ppb[b]; j++) {
                                    h_pairs[p].block_idx_A = 0;
                                    h_pairs[p].block_idx_B = 0;
                                    h_pairs[p].output_offset = temp_off;
                                    h_pairs[p].output_volume = out_vols[b];
                                    temp_off += out_vols[b];
                                    p++;
                                }
                            }
                        }
                        int64_t total_temp = 0;
                        for (int64_t p = 0; p < total_pairs; p++)
                            total_temp += h_pairs[p].output_volume;

                        // Allocate
                        T *d_temp, *d_out;
                        cudaError_t err = cudaMalloc(&d_temp, total_temp * sizeof(T));
                        if (err == cudaErrorMemoryAllocation) {
                            printf("%-14s %7d %9d %7ld %-8s %10s %8s\n",
                                   kernel_names[ki], num_out, ppb, nominal_vol, dists[di],
                                   "OOM", "-");
                            cudaGetLastError();
                            continue;
                        }
                        check_cuda(err, "malloc d_temp");
                        err = cudaMalloc(&d_out, total_out * sizeof(T));
                        if (err == cudaErrorMemoryAllocation) {
                            cudaFree(d_temp); cudaGetLastError();
                            printf("%-14s %7d %9d %7ld %-8s %10s %8s\n",
                                   kernel_names[ki], num_out, ppb, nominal_vol, dists[di],
                                   "OOM", "-");
                            continue;
                        }
                        check_cuda(err, "malloc d_out");

                        // Fill temp with 1s
                        cudaMemsetAsync(d_temp, 0x3f, total_temp * sizeof(T), stream);

                        BenchResult result;

                        {
                            // reduce_noatomic
                            ReductionPlan plan;
                            plan.num_pairs = total_pairs;
                            plan.num_reduce_groups = num_out;

                            plan.h_reduce_dest_offsets = (int64_t*)malloc(num_out * sizeof(int64_t));
                            plan.h_reduce_volumes = (int64_t*)malloc(num_out * sizeof(int64_t));
                            plan.h_reduce_src_ptrs = (int64_t*)malloc((num_out + 1) * sizeof(int64_t));

                            // Count total src entries = total_pairs
                            plan.h_reduce_src_offsets = (int64_t*)malloc(total_pairs * sizeof(int64_t));

                            int64_t p = 0;
                            for (int b = 0; b < num_out; b++) {
                                plan.h_reduce_dest_offsets[b] = h_out_offsets[b];
                                plan.h_reduce_volumes[b] = out_vols[b];
                                plan.h_reduce_src_ptrs[b] = p;
                                for (int j = 0; j < block_ppb[b]; j++) {
                                    plan.h_reduce_src_offsets[p] = h_pairs[p].output_offset;
                                    p++;
                                }
                            }
                            plan.h_reduce_src_ptrs[num_out] = total_pairs;

                            upload_reduction_plan(plan, stream);
                            cudaStreamSynchronize(stream);

                            result = gpu_bench([&]() {
                                cudaMemsetAsync(d_out, 0, total_out * sizeof(T), stream);
                                launch_reduce_noatomic<T, 256, 4>(
                                    d_temp, d_out, plan, stream);
                            }, warmup, repeat, stream);

                            // Free plan
                            free(plan.h_reduce_dest_offsets);
                            free(plan.h_reduce_volumes);
                            free(plan.h_reduce_src_ptrs);
                            free(plan.h_reduce_src_offsets);
                            cudaFree(plan.d_reduce_dest_offsets);
                            cudaFree(plan.d_reduce_volumes);
                            cudaFree(plan.d_reduce_src_offsets);
                            cudaFree(plan.d_reduce_src_ptrs);
                        }

                        // Throughput: read temp + write output
                        double bytes = (double)(total_temp + total_out) * sizeof(T);
                        double gbps = bytes / (result.median_ms * 1e6);

                        printf("%-14s %7d %9d %7ld %-8s %10.3f %8.1f\n",
                               kernel_names[ki], num_out, ppb, nominal_vol, dists[di],
                               result.median_ms, gbps);

                        cudaFree(d_temp);
                        cudaFree(d_out);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generic Permute: Flat vs ND SMEM-Tiled comparison
// ---------------------------------------------------------------------------

// Build ND tile work items for generic permute with (num_I, num_O) collapse.
// I set: trailing source dims {head_rank-1, ..., head_rank-num_I}
// O set: source dims mapping to trailing dest positions
// Batch dims: head dims not in I or O
static void build_nd_gp_tiles(
    int num_I, int num_O,
    int num_blocks, int rank, int head_rank,
    const std::vector<int32_t>& perm,
    const std::vector<int32_t>& inv_perm,
    const std::vector<int64_t>& h_offsets,
    const std::vector<int64_t>& h_dims,
    std::vector<NDPermuteTileWork>& out_tiles)
{
    constexpr int TILE = 32;
    if (head_rank < 2) return;

    // I set: trailing source dims (innermost-first)
    std::vector<int> I_set;
    for (int i = 0; i < num_I; ++i)
        I_set.push_back(head_rank - 1 - i);

    // O set: source dims that map to trailing dest positions (innermost-first)
    std::vector<int> O_set;
    for (int i = 0; i < num_O; ++i)
        O_set.push_back(inv_perm[head_rank - 1 - i]);

    // Batch dims: head dims not in I or O
    std::unordered_set<int> io_set(I_set.begin(), I_set.end());
    io_set.insert(O_set.begin(), O_set.end());
    std::vector<int> batch_dims;
    for (int d = 0; d < head_rank; ++d)
        if (io_set.find(d) == io_set.end()) batch_dims.push_back(d);

    for (int b = 0; b < num_blocks; ++b) {
        const int64_t* dims = &h_dims[b * rank];

        // Only handle tail_vol == 1
        int64_t tail_vol = 1;
        for (int r = head_rank; r < rank; ++r) tail_vol *= dims[r];
        if (tail_vol != 1) continue;

        // Head source strides (row-major)
        std::vector<int64_t> head_src_stride(head_rank);
        head_src_stride[head_rank - 1] = 1;
        for (int d = head_rank - 2; d >= 0; --d)
            head_src_stride[d] = head_src_stride[d + 1] * dims[d + 1];

        // Head dest strides (positional)
        std::vector<int64_t> dst_strides_pos(head_rank);
        dst_strides_pos[head_rank - 1] = 1;
        for (int dd = head_rank - 2; dd >= 0; --dd)
            dst_strides_pos[dd] = dst_strides_pos[dd + 1] * dims[inv_perm[dd + 1]];

        // dst_stride_for_src[sd] = stride at dest position perm[sd]
        std::vector<int64_t> dst_stride_for_src(head_rank);
        for (int sd = 0; sd < head_rank; ++sd)
            dst_stride_for_src[sd] = dst_strides_pos[perm[sd]];

        // Compute I dims (innermost-first in source)
        int32_t I_shape[ND_MAX_COLLAPSE];
        int64_t I_dst_stride[ND_MAX_COLLAPSE];
        int32_t flat_I = 1;
        for (int i = 0; i < ND_MAX_COLLAPSE; ++i) {
            if (i < num_I) {
                I_shape[i] = static_cast<int32_t>(dims[I_set[i]]);
                I_dst_stride[i] = dst_stride_for_src[I_set[i]];
                flat_I *= I_shape[i];
            } else {
                I_shape[i] = 1;
                I_dst_stride[i] = 0;
            }
        }

        // Compute O dims (innermost-first in dest)
        int32_t O_shape[ND_MAX_COLLAPSE];
        int64_t O_src_stride[ND_MAX_COLLAPSE];
        int32_t flat_O = 1;
        for (int i = 0; i < ND_MAX_COLLAPSE; ++i) {
            if (i < num_O) {
                O_shape[i] = static_cast<int32_t>(dims[O_set[i]]);
                O_src_stride[i] = head_src_stride[O_set[i]];
                flat_O *= O_shape[i];
            } else {
                O_shape[i] = 1;
                O_src_stride[i] = 0;
            }
        }

        // Batch dims
        int batch_rank = static_cast<int>(batch_dims.size());
        int batch_elements = 1;
        std::vector<int32_t> batch_shape(batch_rank);
        std::vector<int64_t> batch_src_str(batch_rank);
        std::vector<int64_t> batch_dst_str(batch_rank);
        for (int i = 0; i < batch_rank; ++i) {
            int d = batch_dims[i];
            batch_shape[i] = static_cast<int32_t>(dims[d]);
            batch_src_str[i] = head_src_stride[d];
            batch_dst_str[i] = dst_stride_for_src[d];
            batch_elements *= batch_shape[i];
        }

        // Enumerate (batch, tile_i, tile_o) combinations
        for (int bi = 0; bi < batch_elements; ++bi) {
            int64_t bsrc = 0, bdst = 0;
            int rem = bi;
            for (int i = batch_rank - 1; i >= 0; --i) {
                int coord = rem % batch_shape[i];
                rem /= batch_shape[i];
                bsrc += coord * batch_src_str[i];
                bdst += coord * batch_dst_str[i];
            }

            for (int32_t ti = 0; ti < flat_I; ti += TILE) {
                for (int32_t to = 0; to < flat_O; to += TILE) {
                    NDPermuteTileWork tw{};
                    tw.src_base = h_offsets[b] + bsrc;
                    tw.dst_base = h_offsets[b] + bdst;
                    tw.tile_i = ti;
                    tw.tile_o = to;
                    tw.flat_I_size = flat_I;
                    tw.flat_O_size = flat_O;
                    for (int d = 0; d < ND_MAX_COLLAPSE; ++d) {
                        tw.O_shape[d] = O_shape[d];
                        tw.O_src_stride[d] = O_src_stride[d];
                        tw.I_shape[d] = I_shape[d];
                        tw.I_dst_stride[d] = I_dst_stride[d];
                    }
                    out_tiles.push_back(tw);
                }
            }
        }
    }
}

template <typename T>
static void run_gp_nd_bench(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Generic Permute: Flat vs ND SMEM-Tiled ===\n");
    printf("%-7s %6s %4s %-8s  %2s %2s  %10s %8s  %10s %8s  %s\n",
           "Volume", "Blocks", "Rank", "Dist", "NI", "NO",
           "Flat(ms)", "Flat GB/s", "ND(ms)", "ND GB/s", "Correct");
    printf("%-7s %6s %4s %-8s  %2s %2s  %10s %8s  %10s %8s  %s\n",
           "------", "------", "----", "----", "--", "--",
           "--------", "--------", "------", "-------", "-------");

    int num_blocks_arr[] = {100, 1000, 10000};
    int64_t volume_arr[] = {4096, 16384, 65536};
    int rank_arr[] = {3, 4, 6};
    const char* dists[] = {"uniform", "skewed"};

    for (auto nominal_vol : volume_arr) {
        for (auto rank : rank_arr) {
            for (auto num_blocks : num_blocks_arr) {
                // Use unmoved_tail=0: all dims permuted (tail_vol=1).
                int head_rank = rank;
                if (head_rank < 2) continue;

                // Full reverse permutation (no identity tail dims)
                std::vector<int32_t> h_perm(rank);
                for (int i = 0; i < rank; i++)
                    h_perm[i] = rank - 1 - i;

                std::vector<int32_t> inv_perm(head_rank);
                for (int sd = 0; sd < head_rank; ++sd)
                    inv_perm[h_perm[sd]] = sd;

                for (int di = 0; di < 2; di++) {
                    bool skewed = (di == 1);

                    // Generate volumes and block dims
                    auto nominal_volumes = generate_volumes(num_blocks, nominal_vol, skewed);
                    std::vector<int64_t> h_offsets(num_blocks);
                    std::vector<int64_t> h_dims(num_blocks * rank);
                    std::vector<int64_t> volumes(num_blocks);
                    int64_t total = 0;
                    for (int b = 0; b < num_blocks; b++) {
                        h_offsets[b] = total;
                        auto dims = factor_dims(nominal_volumes[b], rank);
                        for (int r = 0; r < rank; r++)
                            h_dims[b * rank + r] = dims[r];
                        volumes[b] = product(dims);
                        total += volumes[b];
                    }

                    // Allocate device memory
                    T *d_src, *d_dst_flat, *d_dst_nd;
                    cudaError_t err = cudaMalloc(&d_src, total * sizeof(T));
                    if (err == cudaErrorMemoryAllocation) {
                        printf("%7ld %6d %4d %-8s  %2s %2s  %10s %8s  %10s %8s  %s\n",
                               nominal_vol, num_blocks, rank, dists[di],
                               "-", "-", "OOM", "-", "OOM", "-", "-");
                        cudaGetLastError();
                        continue;
                    }
                    check_cuda(err, "malloc src");
                    err = cudaMalloc(&d_dst_flat, total * sizeof(T));
                    if (err != cudaSuccess) {
                        cudaFree(d_src); cudaGetLastError();
                        continue;
                    }
                    err = cudaMalloc(&d_dst_nd, total * sizeof(T));
                    if (err != cudaSuccess) {
                        cudaFree(d_src); cudaFree(d_dst_flat); cudaGetLastError();
                        continue;
                    }

                    // Fill src with deterministic data
                    {
                        std::vector<T> h_src(total);
                        for (int64_t i = 0; i < total; ++i)
                            h_src[i] = static_cast<T>(i % 10000) * static_cast<T>(0.001);
                        cudaMemcpyAsync(d_src, h_src.data(), total * sizeof(T),
                                        cudaMemcpyHostToDevice, stream);
                    }

                    // --- Build flat descs (reference) ---
                    int kl_stride = 4 + 2 * head_rank;
                    std::vector<int64_t> kl_desc_data(num_blocks * kl_stride);
                    std::vector<int64_t> work_per(num_blocks);
                    for (int b = 0; b < num_blocks; b++) {
                        const int64_t* dims = &h_dims[b * rank];
                        int64_t tail_vol = 1;
                        for (int r = head_rank; r < rank; r++) tail_vol *= dims[r];

                        int64_t dst_strides[16];
                        dst_strides[head_rank - 1] = 1;
                        for (int dd = head_rank - 2; dd >= 0; --dd)
                            dst_strides[dd] = dst_strides[dd+1] * dims[inv_perm[dd+1]];

                        int64_t* desc = &kl_desc_data[b * kl_stride];
                        desc[0] = h_offsets[b];
                        desc[1] = h_offsets[b];  // dst_data_offset = src (identity)
                        desc[2] = tail_vol;
                        desc[3] = volumes[b];
                        for (int d = 0; d < head_rank; d++)
                            desc[4 + d] = dims[d];
                        for (int sd = 0; sd < head_rank; sd++)
                            desc[4 + head_rank + sd] = dst_strides[h_perm[sd]];
                        work_per[b] = volumes[b];
                    }

                    auto h_cta = build_cta_work(work_per.data(), num_blocks, 256 * 64, 256);
                    int kl_num_ctas = static_cast<int>(h_cta.size());

                    int64_t *d_kl_descs;
                    CTAWork *d_kl_cta;
                    cudaMalloc(&d_kl_descs, kl_desc_data.size() * sizeof(int64_t));
                    cudaMemcpyAsync(d_kl_descs, kl_desc_data.data(),
                        kl_desc_data.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
                    cudaMalloc(&d_kl_cta, h_cta.size() * sizeof(CTAWork));
                    cudaMemcpyAsync(d_kl_cta, h_cta.data(),
                        h_cta.size() * sizeof(CTAWork),
                        cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);

                    // Run flat kernel once as reference
                    transpose_blocks_generic_permute_batched_kernel<T, 256>
                        <<<kl_num_ctas, 256, 0, stream>>>(
                            d_src, d_dst_flat, d_kl_descs, d_kl_cta,
                            head_rank, kl_stride);
                    cudaStreamSynchronize(stream);

                    double bytes = static_cast<double>(total) * sizeof(T) * 2.0;

                    // Benchmark flat kernel
                    auto flat_result = gpu_bench([&]() {
                        transpose_blocks_generic_permute_batched_kernel<T, 256>
                            <<<kl_num_ctas, 256, 0, stream>>>(
                                d_src, d_dst_flat, d_kl_descs, d_kl_cta,
                                head_rank, kl_stride);
                    }, warmup, repeat, stream);
                    double flat_gbps = bytes / (flat_result.median_ms * 1e6);

                    // Try each valid (num_I, num_O) combination
                    for (int ni = 1; ni <= std::min(ND_MAX_COLLAPSE, head_rank); ++ni) {
                        int i_lo = head_rank - ni;  // lower bound of I set
                        for (int no = 1; no <= std::min(ND_MAX_COLLAPSE, head_rank - ni); ++no) {
                            // Check O set doesn't overlap I set
                            int o_dim = inv_perm[head_rank - no];
                            if (o_dim >= i_lo) continue;
                            // Also check all O dims don't overlap
                            bool overlap = false;
                            for (int oi = 0; oi < no; ++oi) {
                                if (inv_perm[head_rank - 1 - oi] >= i_lo) { overlap = true; break; }
                            }
                            if (overlap) continue;

                            // Build ND tile work items
                            std::vector<NDPermuteTileWork> h_tiles;
                            build_nd_gp_tiles(ni, no, num_blocks, rank, head_rank,
                                              h_perm, inv_perm, h_offsets, h_dims, h_tiles);
                            int tile_num_ctas = static_cast<int>(h_tiles.size());
                            if (tile_num_ctas == 0) continue;

                            NDPermuteTileWork* d_tiles = nullptr;
                            err = cudaMalloc(&d_tiles, tile_num_ctas * sizeof(NDPermuteTileWork));
                            if (err == cudaErrorMemoryAllocation) {
                                cudaGetLastError();
                                printf("%7ld %6d %4d %-8s  %2d %2d  %10.3f %8.1f  %10s %8s  %s\n",
                                       nominal_vol, num_blocks, rank, dists[di],
                                       ni, no, flat_result.median_ms, flat_gbps,
                                       "OOM", "-", "-");
                                continue;
                            }
                            cudaMemcpyAsync(d_tiles, h_tiles.data(),
                                tile_num_ctas * sizeof(NDPermuteTileWork),
                                cudaMemcpyHostToDevice, stream);
                            cudaStreamSynchronize(stream);

                            // Clear and run ND kernel
                            cudaMemsetAsync(d_dst_nd, 0, total * sizeof(T), stream);
                            int nd_grid = std::min(tile_num_ctas, 2048);
                            generic_permute_nd_tile_kernel<T>
                                <<<nd_grid, 256, 0, stream>>>(
                                    d_src, d_dst_nd, d_tiles, tile_num_ctas);
                            cudaStreamSynchronize(stream);

                            // Correctness check against flat
                            std::vector<T> h_flat(total), h_nd(total);
                            cudaMemcpy(h_flat.data(), d_dst_flat, total * sizeof(T),
                                       cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_nd.data(), d_dst_nd, total * sizeof(T),
                                       cudaMemcpyDeviceToHost);
                            bool match = true;
                            for (int64_t i = 0; i < total; ++i) {
                                if (h_flat[i] != h_nd[i]) { match = false; break; }
                            }

                            // Benchmark ND kernel
                            auto nd_result = gpu_bench([&]() {
                                generic_permute_nd_tile_kernel<T>
                                    <<<nd_grid, 256, 0, stream>>>(
                                        d_src, d_dst_nd, d_tiles, tile_num_ctas);
                            }, warmup, repeat, stream);
                            double nd_gbps = bytes / (nd_result.median_ms * 1e6);

                            printf("%7ld %6d %4d %-8s  %2d %2d  %10.3f %8.1f  %10.3f %8.1f  %s\n",
                                   nominal_vol, num_blocks, rank, dists[di],
                                   ni, no, flat_result.median_ms, flat_gbps,
                                   nd_result.median_ms, nd_gbps,
                                   match ? "PASS" : "FAIL");

                            cudaFree(d_tiles);
                        }
                    }

                    cudaFree(d_src);
                    cudaFree(d_dst_flat);
                    cudaFree(d_dst_nd);
                    cudaFree(d_kl_descs);
                    cudaFree(d_kl_cta);
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Hypertile vs Flat generic permute benchmark
// ---------------------------------------------------------------------------

template <typename T>
static void run_hypertile_bench(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Hypertile vs Flat Generic Permute ===\n");
    printf("%-7s %6s %4s %-8s  %3s %5s %7s  %10s %8s  %10s %8s  %s\n",
           "Volume", "Blocks", "Rank", "Dist", "SqR", "TVol", "Tiles",
           "Flat(ms)", "Flat GB/s", "HT(ms)", "HT GB/s", "Correct");
    printf("%-7s %6s %4s %-8s  %3s %5s %7s  %10s %8s  %10s %8s  %s\n",
           "------", "------", "----", "----", "---", "-----", "-------",
           "--------", "--------", "------", "-------", "-------");

    int num_blocks_arr[] = {100, 1000, 10000};
    int64_t volume_arr[] = {4096, 16384, 65536};
    int rank_arr[] = {3, 4, 6};
    const char* dists[] = {"uniform", "skewed"};

    for (auto nominal_vol : volume_arr) {
        for (auto rank : rank_arr) {
            for (auto num_blocks : num_blocks_arr) {
                int head_rank = rank;
                if (head_rank < 2) continue;

                // Full reverse permutation (all dims permuted, tail_vol=1)
                std::vector<int32_t> h_perm(rank);
                for (int i = 0; i < rank; i++)
                    h_perm[i] = rank - 1 - i;

                std::vector<int32_t> inv_perm(head_rank);
                for (int sd = 0; sd < head_rank; ++sd)
                    inv_perm[h_perm[sd]] = sd;

                for (int di = 0; di < 2; di++) {
                    bool skewed = (di == 1);

                    // Generate volumes and block dims
                    auto nominal_volumes = generate_volumes(num_blocks, nominal_vol, skewed);
                    std::vector<int64_t> h_offsets(num_blocks);
                    std::vector<int64_t> h_dims_flat(num_blocks * rank);
                    std::vector<int64_t> volumes(num_blocks);
                    std::vector<int64_t> h_block_volumes(num_blocks);
                    int64_t total = 0;
                    for (int b = 0; b < num_blocks; b++) {
                        h_offsets[b] = total;
                        auto dims = factor_dims(nominal_volumes[b], rank);
                        for (int r = 0; r < rank; r++)
                            h_dims_flat[b * rank + r] = dims[r];
                        volumes[b] = product(dims);
                        h_block_volumes[b] = volumes[b];
                        total += volumes[b];
                    }

                    // Build synthetic GPUSparseInput for hypertile task builder
                    GPUSparseInput<T> gpu;
                    gpu.rank = rank;
                    gpu.num_blocks = num_blocks;
                    gpu.block_offsets = h_offsets;
                    gpu.block_volumes = h_block_volumes;
                    gpu.block_dims.resize(num_blocks * rank);
                    for (int i = 0; i < num_blocks * rank; ++i)
                        gpu.block_dims[i] = static_cast<dim_t>(h_dims_flat[i]);

                    // Build OperandPlan
                    GemmLayoutPlan::OperandPlan op_plan;
                    op_plan.perm = h_perm;
                    op_plan.unmoved_tail_dims = 0;

                    // Build hypertile index + GPU data
                    auto shape_map = build_block_shape_map(gpu.block_dims.data(), num_blocks, rank);
                    auto ht_index = build_hypertile_index<T>(gpu, op_plan, shape_map);
                    auto ht_gpu = build_hypertile_gpu<T>(gpu, ht_index, shape_map, stream);
                    int ht_total_tiles = ht_index.max_tile_volume; // approximate for display
                    { int t = 0; for (auto& sl : ht_gpu.slices) t += sl.total_tiles; ht_total_tiles = t; }

                    if (ht_gpu.slices.empty()) {
                        printf("%7ld %6d %4d %-8s  %3s %5s %7s  %10s %8s  %10s %8s  %s\n",
                               nominal_vol, num_blocks, rank, dists[di],
                               "-", "-", "0", "-", "-", "skip", "-", "-");
                        continue;
                    }

                    int max_sq_rank = ht_index.max_squashed_rank;
                    int max_tile_vol = ht_index.max_tile_volume;

                    // Allocate device memory
                    T *d_src, *d_dst_flat, *d_dst_ht;
                    cudaError_t err = cudaMalloc(&d_src, total * sizeof(T));
                    if (err == cudaErrorMemoryAllocation) {
                        printf("%7ld %6d %4d %-8s  %3d %5d %7d  %10s %8s  %10s %8s  %s\n",
                               nominal_vol, num_blocks, rank, dists[di],
                               max_sq_rank, max_tile_vol, ht_total_tiles,
                               "OOM", "-", "OOM", "-", "-");
                        cudaGetLastError();
                        continue;
                    }
                    check_cuda(err, "malloc src");
                    err = cudaMalloc(&d_dst_flat, total * sizeof(T));
                    if (err != cudaSuccess) {
                        cudaFree(d_src); cudaGetLastError();
                        continue;
                    }
                    err = cudaMalloc(&d_dst_ht, total * sizeof(T));
                    if (err != cudaSuccess) {
                        cudaFree(d_src); cudaFree(d_dst_flat); cudaGetLastError();
                        continue;
                    }

                    // Fill src with deterministic data
                    {
                        std::vector<T> h_src(total);
                        for (int64_t i = 0; i < total; ++i)
                            h_src[i] = static_cast<T>(i % 10000) * static_cast<T>(0.001);
                        cudaMemcpyAsync(d_src, h_src.data(), total * sizeof(T),
                                        cudaMemcpyHostToDevice, stream);
                    }

                    // --- Build flat descs (reference) ---
                    int kl_stride = 4 + 2 * head_rank;
                    std::vector<int64_t> kl_desc_data(num_blocks * kl_stride);
                    std::vector<int64_t> work_per(num_blocks);
                    for (int b = 0; b < num_blocks; b++) {
                        const int64_t* dims = &h_dims_flat[b * rank];
                        int64_t tail_vol = 1;
                        for (int r = head_rank; r < rank; r++) tail_vol *= dims[r];

                        int64_t dst_strides[16];
                        dst_strides[head_rank - 1] = 1;
                        for (int dd = head_rank - 2; dd >= 0; --dd)
                            dst_strides[dd] = dst_strides[dd+1] * dims[inv_perm[dd+1]];

                        int64_t* desc = &kl_desc_data[b * kl_stride];
                        desc[0] = h_offsets[b];
                        desc[1] = h_offsets[b];  // dst_data_offset = src (identity)
                        desc[2] = tail_vol;
                        desc[3] = volumes[b];
                        for (int d = 0; d < head_rank; d++)
                            desc[4 + d] = dims[d];
                        for (int sd = 0; sd < head_rank; sd++)
                            desc[4 + head_rank + sd] = dst_strides[h_perm[sd]];
                        work_per[b] = volumes[b];
                    }

                    auto h_cta = build_cta_work(work_per.data(), num_blocks, 256 * 64, 256);
                    int kl_num_ctas = static_cast<int>(h_cta.size());

                    int64_t *d_kl_descs;
                    CTAWork *d_kl_cta;
                    cudaMalloc(&d_kl_descs, kl_desc_data.size() * sizeof(int64_t));
                    cudaMemcpyAsync(d_kl_descs, kl_desc_data.data(),
                        kl_desc_data.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
                    cudaMalloc(&d_kl_cta, h_cta.size() * sizeof(CTAWork));
                    cudaMemcpyAsync(d_kl_cta, h_cta.data(),
                        h_cta.size() * sizeof(CTAWork),
                        cudaMemcpyHostToDevice, stream);

                    cudaStreamSynchronize(stream);

                    // Run flat kernel once as reference
                    transpose_blocks_generic_permute_batched_kernel<T, 256>
                        <<<kl_num_ctas, 256, 0, stream>>>(
                            d_src, d_dst_flat, d_kl_descs, d_kl_cta,
                            head_rank, kl_stride);
                    cudaStreamSynchronize(stream);

                    double bytes = static_cast<double>(total) * sizeof(T) * 2.0;

                    // Benchmark flat kernel
                    auto flat_result = gpu_bench([&]() {
                        transpose_blocks_generic_permute_batched_kernel<T, 256>
                            <<<kl_num_ctas, 256, 0, stream>>>(
                                d_src, d_dst_flat, d_kl_descs, d_kl_cta,
                                head_rank, kl_stride);
                    }, warmup, repeat, stream);
                    double flat_gbps = bytes / (flat_result.median_ms * 1e6);

                    // Run hypertile kernel once for correctness
                    cudaMemsetAsync(d_dst_ht, 0, total * sizeof(T), stream);
                    constexpr int HT_TV = 1024;
                    constexpr int HT_BS = 256;
                    for (auto& sl : ht_gpu.slices) {
                        launch_hypertile_kernel<T, HT_TV, HT_BS>(
                            d_src, d_dst_ht, sl.d_shapes, sl.d_mems, sl.d_prefix_sum,
                            sl.num_tasks, sl.total_tiles, sl.rank, stream);
                    }
                    cudaStreamSynchronize(stream);

                    // Correctness: compare hypertile output vs flat
                    std::vector<T> h_flat(total), h_ht(total);
                    cudaMemcpy(h_flat.data(), d_dst_flat, total * sizeof(T),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_ht.data(), d_dst_ht, total * sizeof(T),
                               cudaMemcpyDeviceToHost);
                    bool match = true;
                    for (int b = 0; b < num_blocks; ++b) {
                        int64_t off = h_offsets[b];
                        int64_t vol = h_block_volumes[b];
                        for (int64_t i = 0; i < vol; ++i) {
                            if (h_flat[off + i] != h_ht[off + i]) {
                                match = false;
                                break;
                            }
                        }
                        if (!match) break;
                    }

                    // Benchmark hypertile kernel
                    auto ht_result = gpu_bench([&]() {
                        for (auto& sl : ht_gpu.slices) {
                            launch_hypertile_kernel<T, HT_TV, HT_BS>(
                                d_src, d_dst_ht, sl.d_shapes, sl.d_mems, sl.d_prefix_sum,
                                sl.num_tasks, sl.total_tiles, sl.rank, stream);
                        }
                    }, warmup, repeat, stream);

                    // HT bytes: all blocks route to hypertile
                    int64_t ht_elems = 0;
                    for (int b = 0; b < num_blocks; ++b)
                        ht_elems += h_block_volumes[b];
                    double ht_bytes = static_cast<double>(ht_elems) * sizeof(T) * 2.0;
                    double ht_gbps = ht_bytes / (ht_result.median_ms * 1e6);

                    printf("%7ld %6d %4d %-8s  %3d %5d %7d  %10.3f %8.1f  %10.3f %8.1f  %s\n",
                           nominal_vol, num_blocks, rank, dists[di],
                           max_sq_rank, max_tile_vol, ht_total_tiles,
                           flat_result.median_ms, flat_gbps,
                           ht_result.median_ms, ht_gbps,
                           match ? "PASS" : "FAIL");

                    cudaFree(d_src);
                    cudaFree(d_dst_flat);
                    cudaFree(d_dst_ht);
                    cudaFree(d_kl_descs);
                    cudaFree(d_kl_cta);
                    ht_gpu.free_gpu();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hypertile correctness tests — comprehensive permutation & shape coverage
// ---------------------------------------------------------------------------

// Run a single hypertile correctness test case.
// Returns true if passed, false if failed.
template <typename T>
static bool run_single_hypertile_test(
    const char* label,
    int num_blocks,
    int rank,
    const std::vector<int32_t>& perm,
    const std::vector<std::vector<int64_t>>& block_shapes,  // [num_blocks][rank]
    cudaStream_t stream)
{
    // Build GPUSparseInput
    GPUSparseInput<T> gpu;
    gpu.rank = rank;
    gpu.num_blocks = num_blocks;
    gpu.block_offsets.resize(num_blocks);
    gpu.block_volumes.resize(num_blocks);
    gpu.block_dims.resize(num_blocks * rank);

    int64_t total = 0;
    for (int b = 0; b < num_blocks; ++b) {
        gpu.block_offsets[b] = total;
        int64_t vol = 1;
        for (int r = 0; r < rank; ++r) {
            gpu.block_dims[b * rank + r] = static_cast<dim_t>(block_shapes[b][r]);
            vol *= block_shapes[b][r];
        }
        gpu.block_volumes[b] = vol;
        total += vol;
    }

    // Build OperandPlan
    GemmLayoutPlan::OperandPlan op_plan;
    op_plan.perm = perm;
    op_plan.unmoved_tail_dims = 0;

    // Build hypertile index + GPU data
    auto shape_map = build_block_shape_map(gpu.block_dims.data(), num_blocks, rank);
    auto ht_index = build_hypertile_index<T>(gpu, op_plan, shape_map);
    auto ht_gpu = build_hypertile_gpu<T>(gpu, ht_index, shape_map, stream);
    if (ht_gpu.slices.empty()) {
        printf("  %-50s  SKIP (no hypertile tasks)\n", label);
        return true;
    }

    // Compute inverse perm for flat reference
    int head_rank = rank;
    std::vector<int32_t> inv_perm(head_rank);
    for (int sd = 0; sd < head_rank; ++sd)
        inv_perm[perm[sd]] = sd;

    // Allocate device memory
    T *d_src = nullptr, *d_dst_flat = nullptr, *d_dst_ht = nullptr;
    cudaError_t err = cudaMalloc(&d_src, total * sizeof(T));
    if (err == cudaErrorMemoryAllocation) {
        printf("  %-50s  SKIP (OOM: %.1f GB)\n", label,
               (double)total * sizeof(T) / 1e9);
        cudaGetLastError();
        return true;
    }
    check_cuda(err, "malloc src");
    err = cudaMalloc(&d_dst_flat, total * sizeof(T));
    if (err != cudaSuccess) { cudaFree(d_src); cudaGetLastError();
        printf("  %-50s  SKIP (OOM)\n", label); return true; }
    err = cudaMalloc(&d_dst_ht, total * sizeof(T));
    if (err != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst_flat); cudaGetLastError();
        printf("  %-50s  SKIP (OOM)\n", label); return true; }

    // Fill src with deterministic data
    {
        std::vector<T> h_src(total);
        for (int64_t i = 0; i < total; ++i)
            h_src[i] = static_cast<T>(i % 10000) * static_cast<T>(0.001);
        cudaMemcpyAsync(d_src, h_src.data(), total * sizeof(T),
                        cudaMemcpyHostToDevice, stream);
    }

    // --- Build flat generic permute descs (reference) ---
    int kl_stride = 4 + 2 * head_rank;
    std::vector<int64_t> kl_desc_data(num_blocks * kl_stride);
    std::vector<int64_t> work_per(num_blocks);
    for (int b = 0; b < num_blocks; b++) {
        int64_t dst_strides[16];
        dst_strides[head_rank - 1] = 1;
        for (int dd = head_rank - 2; dd >= 0; --dd)
            dst_strides[dd] = dst_strides[dd+1] * block_shapes[b][inv_perm[dd+1]];

        int64_t* desc = &kl_desc_data[b * kl_stride];
        desc[0] = gpu.block_offsets[b];
        desc[1] = gpu.block_offsets[b];  // dst_data_offset = src (identity)
        desc[2] = 1;  // tail_vol = 1 (unmoved_tail_dims = 0)
        desc[3] = gpu.block_volumes[b];
        for (int d = 0; d < head_rank; d++)
            desc[4 + d] = block_shapes[b][d];
        for (int sd = 0; sd < head_rank; sd++)
            desc[4 + head_rank + sd] = dst_strides[perm[sd]];
        work_per[b] = gpu.block_volumes[b];
    }

    auto h_cta = build_cta_work(work_per.data(), num_blocks, 256 * 64, 256);
    int kl_num_ctas = static_cast<int>(h_cta.size());

    int64_t *d_kl_descs;
    CTAWork *d_kl_cta;
    cudaMalloc(&d_kl_descs, kl_desc_data.size() * sizeof(int64_t));
    cudaMemcpyAsync(d_kl_descs, kl_desc_data.data(),
        kl_desc_data.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_kl_cta, h_cta.size() * sizeof(CTAWork));
    cudaMemcpyAsync(d_kl_cta, h_cta.data(),
        h_cta.size() * sizeof(CTAWork), cudaMemcpyHostToDevice, stream);

    // Run flat reference
    transpose_blocks_generic_permute_batched_kernel<T, 256>
        <<<kl_num_ctas, 256, 0, stream>>>(
            d_src, d_dst_flat, d_kl_descs, d_kl_cta, head_rank, kl_stride);
    cudaStreamSynchronize(stream);

    // --- Run hypertile ---
    cudaMemsetAsync(d_dst_ht, 0, total * sizeof(T), stream);
    constexpr int HT_TV = 1024;
    constexpr int HT_BS = 256;
    for (auto& sl : ht_gpu.slices) {
        launch_hypertile_kernel<T, HT_TV, HT_BS>(
            d_src, d_dst_ht, sl.d_shapes, sl.d_mems, sl.d_prefix_sum,
            sl.num_tasks, sl.total_tiles, sl.rank, stream);
    }
    cudaStreamSynchronize(stream);

    // --- Compare outputs ---
    std::vector<T> h_flat(total), h_ht(total);
    cudaMemcpy(h_flat.data(), d_dst_flat, total * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ht.data(), d_dst_ht, total * sizeof(T), cudaMemcpyDeviceToHost);

    bool match = true;
    int fail_block = -1;
    int64_t fail_idx = -1;
    for (int b = 0; b < num_blocks; ++b) {
        int64_t off = gpu.block_offsets[b];
        int64_t vol = gpu.block_volumes[b];
        for (int64_t i = 0; i < vol; ++i) {
            if (h_flat[off + i] != h_ht[off + i]) {
                match = false;
                fail_block = b;
                fail_idx = i;
                break;
            }
        }
        if (!match) break;
    }

    if (match) {
        printf("  %-50s  PASS\n", label);
    } else {
        printf("  %-50s  FAIL (block %d, elem %ld: flat=%.6f ht=%.6f)\n",
               label, fail_block, (long)fail_idx,
               (double)h_flat[gpu.block_offsets[fail_block] + fail_idx],
               (double)h_ht[gpu.block_offsets[fail_block] + fail_idx]);
    }

    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst_flat);
    cudaFree(d_dst_ht);
    cudaFree(d_kl_descs);
    cudaFree(d_kl_cta);
    ht_gpu.free_gpu();

    return match;
}

// Helper: make uniform blocks with given shape
static std::vector<std::vector<int64_t>> make_uniform_blocks(
    int num_blocks, const std::vector<int64_t>& shape) {
    return std::vector<std::vector<int64_t>>(num_blocks, shape);
}

// Helper: generate a seeded pseudo-random permutation
static std::vector<int32_t> random_perm(int rank, uint32_t seed) {
    std::vector<int32_t> p(rank);
    std::iota(p.begin(), p.end(), 0);
    // Fisher-Yates with simple LCG
    uint32_t s = seed;
    for (int i = rank - 1; i > 0; --i) {
        s = s * 1103515245u + 12345u;
        int j = static_cast<int>(s % (i + 1));
        std::swap(p[i], p[j]);
    }
    return p;
}

// Check if permutation is identity
static bool is_identity_perm(const std::vector<int32_t>& p) {
    for (int i = 0; i < (int)p.size(); ++i)
        if (p[i] != i) return false;
    return true;
}

template <typename T>
static void run_hypertile_correctness_test(cudaStream_t stream) {
    printf("\n=== Hypertile Correctness Tests ===\n");

    int total_tests = 0, total_pass = 0;
    auto run = [&](const char* label, int nb, int rank,
                   const std::vector<int32_t>& perm,
                   const std::vector<std::vector<int64_t>>& shapes) {
        total_tests++;
        if (run_single_hypertile_test<T>(label, nb, rank, perm, shapes, stream))
            total_pass++;
    };

    // =====================================================================
    // Category 1: Arbitrary permutations, moderate blocks
    // =====================================================================
    printf("\n--- Category 1: Arbitrary permutations ---\n");

    // Rank 2: only one non-identity perm [1,0]
    run("R2 reverse 100x 50x80", 100, 2, {1, 0},
        make_uniform_blocks(100, {50, 80}));

    // Rank 3
    run("R3 reverse 100x 16x16x16", 100, 3, {2, 1, 0},
        make_uniform_blocks(100, {16, 16, 16}));
    run("R3 cyclic [1,2,0] 100x 16x16x16", 100, 3, {1, 2, 0},
        make_uniform_blocks(100, {16, 16, 16}));
    run("R3 swap01 [1,0,2] 100x 16x16x16", 100, 3, {1, 0, 2},
        make_uniform_blocks(100, {16, 16, 16}));
    run("R3 swap12 [0,2,1] 100x 16x16x16", 100, 3, {0, 2, 1},
        make_uniform_blocks(100, {16, 16, 16}));

    // Rank 4
    run("R4 reverse 50x 8x8x8x8", 50, 4, {3, 2, 1, 0},
        make_uniform_blocks(50, {8, 8, 8, 8}));
    run("R4 cyclic [1,2,3,0] 50x 8x8x8x8", 50, 4, {1, 2, 3, 0},
        make_uniform_blocks(50, {8, 8, 8, 8}));
    run("R4 swap02 [2,1,0,3] 50x 8x8x8x8", 50, 4, {2, 1, 0, 3},
        make_uniform_blocks(50, {8, 8, 8, 8}));
    run("R4 swap13 [0,3,2,1] 50x 8x8x8x8", 50, 4, {0, 3, 2, 1},
        make_uniform_blocks(50, {8, 8, 8, 8}));

    // Rank 5 (physics-tensor-like)
    run("R5 reverse 20x 4x4x4x4x4", 20, 5, {4, 3, 2, 1, 0},
        make_uniform_blocks(20, {4, 4, 4, 4, 4}));
    run("R5 cyclic [1,2,3,4,0] 20x 4x4x4x4x4", 20, 5, {1, 2, 3, 4, 0},
        make_uniform_blocks(20, {4, 4, 4, 4, 4}));
    run("R5 [3,0,4,1,2] 20x 4x4x4x4x4", 20, 5, {3, 0, 4, 1, 2},
        make_uniform_blocks(20, {4, 4, 4, 4, 4}));

    // Rank 6
    run("R6 reverse 10x 4x3x3x3x3x4", 10, 6, {5, 4, 3, 2, 1, 0},
        make_uniform_blocks(10, {4, 3, 3, 3, 3, 4}));
    run("R6 cyclic [1,2,3,4,5,0] 10x 4x3x3x3x3x4", 10, 6, {1, 2, 3, 4, 5, 0},
        make_uniform_blocks(10, {4, 3, 3, 3, 3, 4}));

    // Rank 7
    run("R7 reverse 5x 3x3x3x3x3x3x3", 5, 7, {6, 5, 4, 3, 2, 1, 0},
        make_uniform_blocks(5, {3, 3, 3, 3, 3, 3, 3}));

    // Random permutations (seeded)
    for (int rank : {3, 4, 5, 6}) {
        for (uint32_t seed = 1; seed <= 5; ++seed) {
            auto perm = random_perm(rank, seed * 100 + rank);
            if (is_identity_perm(perm)) continue;
            char label[128];
            snprintf(label, sizeof(label), "R%d random(seed=%u) 50x uniform",
                     rank, seed * 100 + rank);
            auto shape = factor_dims(rank <= 4 ? 4096 : 1024, rank);
            run(label, 50, rank, perm, make_uniform_blocks(50, shape));
        }
    }

    // =====================================================================
    // Category 2: Block shape variety
    // =====================================================================
    printf("\n--- Category 2: Block shape variety ---\n");

    // Highly rectangular
    run("R2 reverse 50x 2x50000", 50, 2, {1, 0},
        make_uniform_blocks(50, {2, 50000}));
    run("R2 reverse 50x 50000x2", 50, 2, {1, 0},
        make_uniform_blocks(50, {50000, 2}));
    run("R3 reverse 50x 2x2x10000", 50, 3, {2, 1, 0},
        make_uniform_blocks(50, {2, 2, 10000}));
    run("R3 reverse 50x 10000x2x2", 50, 3, {2, 1, 0},
        make_uniform_blocks(50, {10000, 2, 2}));

    // Mixed sizes: some blocks < 1024 (will be skipped by hypertile), some large
    {
        std::vector<std::vector<int64_t>> mixed(100);
        for (int b = 0; b < 100; ++b) {
            if (b % 3 == 0)
                mixed[b] = {4, 4, 4};  // 64 elems, below 1024 threshold
            else if (b % 3 == 1)
                mixed[b] = {16, 16, 16};  // 4096
            else
                mixed[b] = {32, 32, 8};  // 8192
        }
        run("R3 mixed small+large 100x", 100, 3, {2, 1, 0}, mixed);
    }

    // Single block, larger volume
    run("R3 reverse 1x 100x100x100", 1, 3, {2, 1, 0},
        make_uniform_blocks(1, {100, 100, 100}));
    run("R4 reverse 1x 32x32x32x32", 1, 4, {3, 2, 1, 0},
        make_uniform_blocks(1, {32, 32, 32, 32}));
    run("R5 [3,0,4,1,2] 1x 8x8x8x8x8", 1, 5, {3, 0, 4, 1, 2},
        make_uniform_blocks(1, {8, 8, 8, 8, 8}));

    // =====================================================================
    // Category 3: Single large block (>2G elements) — uint32_t overflow test
    // =====================================================================
    printf("\n--- Category 3: Large block (>2G element) overflow tests ---\n");
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        // Each test needs ~3x buffers (src + dst_flat + dst_ht)
        // Shape [2, 1100000000] ≈ 2.2G elems → 2.2G * 4 * 3 ≈ 26.4 GB for f32
        // Shape [2048, 1024, 1024] ≈ 2.15G → ~25.8 GB for f32
        // Shape [16, 138000000] ≈ 2.2G → ~26.4 GB

        // For 20GB GPUs, these will all be skipped. Test with shapes that
        // fit but still stress large offsets.
        int64_t max_elems_per_buf = static_cast<int64_t>(free_mem / sizeof(T)) / 4;
        // /4 for safety margin (3 buffers + metadata)

        if (max_elems_per_buf > static_cast<int64_t>(2200000000LL)) {
            // True >2G tests
            run("R2 [1,0] 1x 16x138000000 (>2G overflow)", 1, 2, {1, 0},
                make_uniform_blocks(1, {16, 138000000}));
            run("R2 [1,0] 1x 2x1100000000 (>2G stride-safe)", 1, 2, {1, 0},
                make_uniform_blocks(1, {2, 1100000000}));
            run("R3 [2,1,0] 1x 2048x1024x1024 (>2G R3)", 1, 3, {2, 1, 0},
                make_uniform_blocks(1, {2048, 1024, 1024}));
        } else {
            printf("  (>2G tests skipped: need ~26 GB free, have %.1f GB)\n",
                   (double)free_mem / 1e9);

            // Fallback: largest block that fits, to stress large-offset paths
            // Use ~500M elements if possible (2GB for f32)
            int64_t target = std::min(max_elems_per_buf, (int64_t)500000000LL);
            if (target >= 1024) {
                {
                    int64_t N = target / 16;
                    if (N >= 1) {
                        char label[128];
                        snprintf(label, sizeof(label),
                                 "R2 [1,0] 1x 16x%ld (large fallback)", (long)N);
                        run(label, 1, 2, {1, 0}, make_uniform_blocks(1, {16, N}));
                    }
                }
                {
                    int64_t side = static_cast<int64_t>(std::cbrt((double)target));
                    if (side >= 2) {
                        char label[128];
                        snprintf(label, sizeof(label),
                                 "R3 [2,1,0] 1x %ldx%ldx%ld (large fallback)",
                                 (long)side, (long)side, (long)side);
                        run(label, 1, 3, {2, 1, 0},
                            make_uniform_blocks(1, {side, side, side}));
                    }
                }
            }
        }
    }

    // =====================================================================
    // Category 4: Real-data-inspired shapes with various perms
    // =====================================================================
    printf("\n--- Category 4: Physics-tensor-inspired shapes ---\n");

    // nf-like: few large blocks, rank 5
    {
        std::vector<std::vector<int64_t>> nf_shapes = {
            {2, 13, 13, 13, 6},   // vol ~13182
            {2, 11, 11, 11, 8},   // vol ~21296
            {1, 15, 15, 15, 4},   // vol ~13500
        };
        for (auto& perm : std::vector<std::vector<int32_t>>{
                {4, 3, 2, 1, 0}, {1, 2, 3, 4, 0}, {0, 4, 1, 2, 3}, {3, 0, 4, 1, 2}}) {
            if (is_identity_perm(perm)) continue;
            char label[128];
            snprintf(label, sizeof(label),
                     "R5 nf-like [%d,%d,%d,%d,%d] 3 blocks",
                     perm[0], perm[1], perm[2], perm[3], perm[4]);
            run(label, 3, 5, perm, nf_shapes);
        }
    }

    // kysznf-like: many small blocks, rank 6
    {
        std::vector<std::vector<int64_t>> ky_shapes(200);
        for (int b = 0; b < 200; ++b) {
            if (b % 4 == 0)
                ky_shapes[b] = {2, 3, 3, 3, 3, 6};   // 972
            else if (b % 4 == 1)
                ky_shapes[b] = {1, 4, 4, 4, 4, 4};   // 1024
            else if (b % 4 == 2)
                ky_shapes[b] = {3, 3, 3, 3, 3, 8};   // 1944
            else
                ky_shapes[b] = {2, 5, 5, 5, 5, 2};   // 2500
        }
        for (auto& perm : std::vector<std::vector<int32_t>>{
                {5, 4, 3, 2, 1, 0}, {1, 2, 3, 4, 5, 0}, {0, 5, 1, 4, 2, 3}}) {
            if (is_identity_perm(perm)) continue;
            char label[128];
            snprintf(label, sizeof(label),
                     "R6 ky-like [%d,%d,%d,%d,%d,%d] 200 blocks",
                     perm[0], perm[1], perm[2], perm[3], perm[4], perm[5]);
            run(label, 200, 6, perm, ky_shapes);
        }
    }

    // Mixed large+small with various perms (rank 4)
    {
        std::vector<std::vector<int64_t>> mixed(50);
        for (int b = 0; b < 50; ++b) {
            if (b < 5)
                mixed[b] = {32, 32, 32, 4};  // 131072
            else if (b < 15)
                mixed[b] = {8, 8, 8, 8};     // 4096
            else
                mixed[b] = {4, 4, 4, 4};     // 256 (below threshold)
        }
        for (auto& perm : std::vector<std::vector<int32_t>>{
                {3, 2, 1, 0}, {1, 2, 3, 0}, {0, 3, 1, 2}, {2, 0, 3, 1}}) {
            if (is_identity_perm(perm)) continue;
            char label[128];
            snprintf(label, sizeof(label),
                     "R4 mixed [%d,%d,%d,%d] 50 blocks",
                     perm[0], perm[1], perm[2], perm[3]);
            run(label, 50, 4, perm, mixed);
        }
    }

    printf("\n=== Hypertile Correctness: %d/%d passed ===\n", total_pass, total_tests);
}

// ---------------------------------------------------------------------------
// Fuzz transpose: random (rank, perm, block shapes) → hypertile vs flat
// ---------------------------------------------------------------------------

template <typename T>
static void run_fuzz_transpose(uint64_t base_seed, int count, bool stop_on_fail,
                               bool verbose, cudaStream_t stream) {
    printf("Fuzz transpose: base_seed=%lu, count=%d, dtype=%s\n\n",
           (unsigned long)base_seed, count,
           std::is_same<T, double>::value ? "f64" : "f32");

    int passed = 0, failed = 0, skipped = 0;

    for (int i = 0; i < count || count == 0; ++i) {
        uint64_t seed = base_seed + static_cast<uint64_t>(i);
        std::mt19937 rng(seed);

        // Random rank 2..7
        int rank = std::uniform_int_distribution<int>(2, 7)(rng);

        // Random non-identity permutation
        std::vector<int32_t> perm(rank);
        std::iota(perm.begin(), perm.end(), 0);
        // Fisher-Yates, retry if identity
        for (int attempt = 0; attempt < 10; ++attempt) {
            for (int j = rank - 1; j > 0; --j) {
                int k = std::uniform_int_distribution<int>(0, j)(rng);
                std::swap(perm[j], perm[k]);
            }
            if (!is_identity_perm(perm)) break;
        }
        if (is_identity_perm(perm)) continue;

        // Random number of blocks 1..200
        int num_blocks = std::uniform_int_distribution<int>(1, 200)(rng);

        // Random block shapes
        // Each dim: 2..100, but constrain volume to [1024, 500000] per block
        std::vector<std::vector<int64_t>> shapes(num_blocks);
        bool any_above_threshold = false;
        for (int b = 0; b < num_blocks; ++b) {
            // Pick dims
            std::vector<int64_t> dims(rank);
            for (int d = 0; d < rank; ++d)
                dims[d] = std::uniform_int_distribution<int64_t>(2, 30)(rng);
            // Optionally make some blocks highly rectangular
            if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.2) {
                int big_dim = std::uniform_int_distribution<int>(0, rank - 1)(rng);
                dims[big_dim] *= std::uniform_int_distribution<int64_t>(10, 100)(rng);
            }
            // Check volume isn't absurdly large (keep total mem reasonable)
            int64_t vol = 1;
            for (auto d : dims) vol *= d;
            // Cap per-block volume at 1M elements
            if (vol > 1000000) {
                double scale = std::pow(1000000.0 / static_cast<double>(vol), 1.0 / rank);
                for (int d = 0; d < rank; ++d)
                    dims[d] = std::max((int64_t)2, static_cast<int64_t>(static_cast<double>(dims[d]) * scale));
                vol = 1;
                for (auto d : dims) vol *= d;
            }
            shapes[b] = dims;
            if (vol >= 1024) any_above_threshold = true;
        }

        // Skip if all blocks below hypertile threshold
        if (!any_above_threshold) {
            if (verbose)
                printf("[%d] seed=%lu R%d %dblk → SKIP (all below threshold)\n",
                       i, (unsigned long)seed, rank, num_blocks);
            ++skipped;
            continue;
        }

        // Check total memory needed (3 buffers)
        int64_t total = 0;
        for (int b = 0; b < num_blocks; ++b) {
            int64_t vol = 1;
            for (int d = 0; d < rank; ++d) vol *= shapes[b][d];
            total += vol;
        }
        int64_t mem_needed = total * static_cast<int64_t>(sizeof(T)) * 3;
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (mem_needed > static_cast<int64_t>(static_cast<double>(free_mem) * 0.8)) {
            if (verbose)
                printf("[%d] seed=%lu R%d %dblk → SKIP (%.1f GB needed)\n",
                       i, (unsigned long)seed, rank, num_blocks,
                       (double)mem_needed / 1e9);
            ++skipped;
            continue;
        }

        // Format perm string for label
        char perm_str[64] = {};
        int pos = 0;
        pos += snprintf(perm_str + pos, sizeof(perm_str) - pos, "[");
        for (int d = 0; d < rank; ++d)
            pos += snprintf(perm_str + pos, sizeof(perm_str) - pos,
                            "%s%d", d ? "," : "", perm[d]);
        snprintf(perm_str + pos, sizeof(perm_str) - pos, "]");

        char label[128];
        snprintf(label, sizeof(label), "R%d %s %dblk", rank, perm_str, num_blocks);

        bool ok = run_single_hypertile_test<T>(label, num_blocks, rank, perm,
                                                shapes, stream);
        if (ok) {
            ++passed;
        } else {
            ++failed;
            printf("  ^^^ Reproduce: --fuzz-transpose --seed %lu --count 1 --verbose\n",
                   (unsigned long)seed);
            if (stop_on_fail) break;
        }

        // Periodic summary
        if ((i + 1) % 100 == 0)
            printf("  ... %d passed, %d failed, %d skipped after %d iterations\n",
                   passed, failed, skipped, i + 1);
    }

    printf("\nFuzz transpose summary: %d passed, %d failed, %d skipped "
           "(seeds %lu–%lu)\n",
           passed, failed, skipped,
           (unsigned long)base_seed,
           (unsigned long)(base_seed + (count > 0 ? count - 1 : 0)));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

template <typename T>
static void run_all_benches(int warmup, int repeat, const std::string& filter, cudaStream_t stream) {
    bool do_transpose = filter.empty() || filter.find("transpose") != std::string::npos;
    bool do_accum = filter.empty() || filter.find("accum") != std::string::npos;
    bool do_gp_nd = filter.find("gp_nd") != std::string::npos;
    bool do_hypertile = filter.find("hypertile") != std::string::npos;
    bool do_hypertile_test = filter.find("hypertile_test") != std::string::npos;

    if (do_transpose) run_transpose_bench<T>(warmup, repeat, stream);
    if (do_gp_nd) run_gp_nd_bench<T>(warmup, repeat, stream);
    if (do_hypertile_test) run_hypertile_correctness_test<T>(stream);
    else if (do_hypertile) run_hypertile_bench<T>(warmup, repeat, stream);
    if (do_accum) run_accum_bench<T>(warmup, repeat, stream);
}

int main(int argc, char** argv) {
    int warmup = 3, repeat = 5;
    std::string filter;
    bool use_f64 = false;
    bool fuzz_transpose = false;
    uint64_t fuzz_seed = 0;
    int fuzz_count = 1000;
    bool fuzz_stop_on_fail = false;
    bool fuzz_verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)
            repeat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc)
            filter = argv[++i];
        else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "f64" || v == "float64") use_f64 = true;
        }
        else if (strcmp(argv[i], "--fuzz-transpose") == 0)
            fuzz_transpose = true;
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            fuzz_seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            fuzz_count = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stop-on-fail") == 0)
            fuzz_stop_on_fail = true;
        else if (strcmp(argv[i], "--verbose") == 0)
            fuzz_verbose = true;
        else {
            fprintf(stderr, "Usage: %s [--warmup N] [--repeat N] [--filter <name>] [--dtype f32|f64]\n"
                    "       %s --fuzz-transpose [--seed N] [--count N] [--stop-on-fail] [--verbose] [--dtype f32|f64]\n",
                    argv[0], argv[0]);
            return 1;
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (fuzz_transpose) {
        if (fuzz_seed == 0)
            fuzz_seed = static_cast<uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if (use_f64)
            run_fuzz_transpose<double>(fuzz_seed, fuzz_count, fuzz_stop_on_fail, fuzz_verbose, stream);
        else
            run_fuzz_transpose<float>(fuzz_seed, fuzz_count, fuzz_stop_on_fail, fuzz_verbose, stream);
    } else {
        printf("Synthetic Kernel Benchmarks (warmup=%d, repeat=%d, dtype=%s)\n",
               warmup, repeat, use_f64 ? "f64" : "f32");

        if (use_f64)
            run_all_benches<double>(warmup, repeat, filter, stream);
        else
            run_all_benches<float>(warmup, repeat, filter, stream);
    }

    cudaStreamDestroy(stream);
    return 0;
}
