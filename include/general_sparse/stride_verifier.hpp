#pragma once

#include "gpu_sparse_input.hpp"
#include "gpu_sparse_device.cuh"
#include "general_sparse_executor.cuh"
#include "cutensor_block_sparse.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

// Stride-based verification for the GPU pipeline.
// Compares pipeline output (internal order, row-major) against cuTENSOR reference
// (output order, column-major) using stride arithmetic instead of data transposition.
namespace bsptc {

struct StrideVerificationResult {
    double max_abs;
    double max_rel;
    int64_t num_compared;
    float cutensor_ms;
    bool passed;
};

// Compute section offsets for each dimension of the output in internal order.
// section_offsets[dim][block_idx] = cumulative offset for that block index.
template <typename T>
inline std::vector<std::vector<int64_t>> compute_internal_section_offsets(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config)
{
    std::vector<std::vector<int64_t>> offsets;

    // Helper to get block sizes for a dimension
    auto get_block_sizes = [](const GPUSparseInput<T>& gpu, int32_t dim_pos) {
        int32_t start = gpu.block_sizes_offsets[dim_pos];
        int32_t end = (dim_pos + 1 < gpu.rank)
                      ? gpu.block_sizes_offsets[dim_pos + 1]
                      : static_cast<int32_t>(gpu.block_sizes_flat.size());
        std::vector<dim_t> sizes;
        for (int32_t i = start; i < end; ++i) {
            sizes.push_back(gpu.block_sizes_flat[i]);
        }
        return sizes;
    };

    // Internal order: batch, free_A, free_B
    for (int32_t i = 0; i < config.num_batch; ++i) {
        auto sizes = get_block_sizes(A_gpu, config.batch_pos_A[i]);
        std::vector<int64_t> sec(sizes.size(), 0);
        int64_t acc = 0;
        for (size_t j = 0; j < sizes.size(); ++j) {
            sec[j] = acc;
            acc += sizes[j];
        }
        offsets.push_back(sec);
    }
    for (int32_t i = 0; i < config.num_free_A; ++i) {
        auto sizes = get_block_sizes(A_gpu, config.free_pos_A[i]);
        std::vector<int64_t> sec(sizes.size(), 0);
        int64_t acc = 0;
        for (size_t j = 0; j < sizes.size(); ++j) {
            sec[j] = acc;
            acc += sizes[j];
        }
        offsets.push_back(sec);
    }
    for (int32_t i = 0; i < config.num_free_B; ++i) {
        auto sizes = get_block_sizes(B_gpu, config.free_pos_B[i]);
        std::vector<int64_t> sec(sizes.size(), 0);
        int64_t acc = 0;
        for (size_t j = 0; j < sizes.size(); ++j) {
            sec[j] = acc;
            acc += sizes[j];
        }
        offsets.push_back(sec);
    }

    return offsets;
}

// Verify pipeline output directly against cuTENSOR block-sparse output.
// Compares block-by-block: checks for missing/extra blocks and element-level diffs.
// Pipeline blocks are internal order (row-major), cuTENSOR blocks are output order (col-major).
template <typename T>
StrideVerificationResult verify_against_block_sparse(
    const GeneralSparseResult<T>& pipeline_result,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config,
    const CutensorBlockSparseResult<T>& ref,
    double atol = 1e-5,
    double rtol = 1e-5)
{
    StrideVerificationResult vr;
    vr.max_abs = 0.0;
    vr.max_rel = 0.0;
    vr.num_compared = 0;
    vr.cutensor_ms = ref.kernel_ms;
    vr.passed = true;

    const dim_t rank = config.output_rank;
    if (rank == 0) return vr;

    // Copy pipeline output from device
    std::vector<T> host_data(pipeline_result.total_output_elements);
    if (pipeline_result.total_output_elements > 0 && pipeline_result.d_output) {
        check_cuda(cudaMemcpy(host_data.data(), pipeline_result.d_output,
                              pipeline_result.total_output_elements * sizeof(T),
                              cudaMemcpyDeviceToHost), "memcpy result for block-sparse verify");
    }

    // Section offsets in internal order (needed for block dim lookups)
    auto sec_offsets = compute_internal_section_offsets(A_gpu, B_gpu, config);

    // Helper to get block size for a given internal dimension and block index
    auto get_block_size = [](const GPUSparseInput<T>& gpu, int32_t dim_pos, int32_t block_idx) -> dim_t {
        return gpu.block_sizes_flat[gpu.block_sizes_offsets[dim_pos] + block_idx];
    };

    // Build map: output-order coord → cuTENSOR block index
    std::map<std::vector<int64_t>, size_t> ref_coord_map;
    for (size_t i = 0; i < ref.block_coords.size(); ++i)
        ref_coord_map[ref.block_coords[i]] = i;

    // Track which cuTENSOR blocks are matched
    std::vector<bool> ref_matched(ref.block_coords.size(), false);

    // For each pipeline block, find the matching cuTENSOR block and compare
    double global_max_abs = 0.0;
    double global_max_rel = 0.0;
    int64_t global_compared = 0;
    bool global_passed = true;

    for (int64_t blk = 0; blk < pipeline_result.num_output_blocks; ++blk) {
        const int32_t* block_coord_internal = &pipeline_result.output_block_coords[blk * rank];
        const int64_t src_offset = pipeline_result.output_offsets[blk];
        const int64_t volume = pipeline_result.output_volumes[blk];

        // Convert internal-order coord to output-order coord
        std::vector<int64_t> out_coord(rank);
        for (int32_t d = 0; d < rank; ++d)
            out_coord[config.internal_to_output[d]] = block_coord_internal[d];

        // Look up in cuTENSOR map
        auto it = ref_coord_map.find(out_coord);
        if (it == ref_coord_map.end()) {
            // Pipeline produced a block not in cuTENSOR output — check if all zeros
            bool all_zero = true;
            for (int64_t e = 0; e < volume; ++e) {
                if (static_cast<double>(std::abs(host_data[src_offset + e])) > atol) {
                    all_zero = false;
                    break;
                }
            }
            if (!all_zero) {
                std::cerr << "verify_against_block_sparse: pipeline block [";
                for (int32_t d = 0; d < rank; ++d) {
                    if (d) std::cerr << ",";
                    std::cerr << out_coord[d];
                }
                std::cerr << "] not in cuTENSOR output and is non-zero\n";
                global_passed = false;
            }
            continue;
        }

        size_t ref_idx = it->second;
        ref_matched[ref_idx] = true;

        // Get block dims in internal order
        dim_t block_dims[MAX_RANK];
        int32_t pos = 0;
        for (int32_t i = 0; i < config.num_batch; ++i, ++pos)
            block_dims[pos] = get_block_size(A_gpu, config.batch_pos_A[i], block_coord_internal[pos]);
        for (int32_t i = 0; i < config.num_free_A; ++i, ++pos)
            block_dims[pos] = get_block_size(A_gpu, config.free_pos_A[i], block_coord_internal[pos]);
        for (int32_t i = 0; i < config.num_free_B; ++i, ++pos)
            block_dims[pos] = get_block_size(B_gpu, config.free_pos_B[i], block_coord_internal[pos]);

        // Compute output-order block shape and col-major strides for the ref block
        dim_t out_block_dims[MAX_RANK];
        for (int32_t d = 0; d < rank; ++d)
            out_block_dims[config.internal_to_output[d]] = block_dims[d];

        int64_t ref_cm_strides[MAX_RANK];
        ref_cm_strides[0] = 1;
        for (int32_t d = 1; d < rank; ++d)
            ref_cm_strides[d] = ref_cm_strides[d - 1] * out_block_dims[d - 1];

        // Precompute: ref_cm_stride remapped to internal order
        int64_t ref_cm_stride_internal[MAX_RANK];
        for (int32_t d = 0; d < rank; ++d)
            ref_cm_stride_internal[d] = ref_cm_strides[config.internal_to_output[d]];

        const T* ref_block = ref.block_data.data() + ref.block_offsets[ref_idx];

        // Iterate over elements in row-major (internal order)
        int64_t local_coord[MAX_RANK] = {};
        double blk_max_abs = 0.0;
        double blk_max_rel = 0.0;
        bool blk_passed = true;

        for (int64_t elem = 0; elem < volume; ++elem) {
            // Compute col-major index in ref block
            int64_t ref_off = 0;
            for (int32_t d = 0; d < rank; ++d)
                ref_off += local_coord[d] * ref_cm_stride_internal[d];

            double a = static_cast<double>(ref_block[ref_off]);
            double b = static_cast<double>(host_data[src_offset + elem]);
            double diff = std::abs(a - b);
            blk_max_abs = std::max(blk_max_abs, diff);
            double denom = std::max(1e-8, std::abs(a));
            blk_max_rel = std::max(blk_max_rel, diff / denom);

            if (diff > atol + rtol * std::abs(a))
                blk_passed = false;

            // Increment local coordinates (row-major: last dimension fastest)
            for (int32_t d = rank - 1; d >= 0; --d) {
                if (++local_coord[d] < block_dims[d]) break;
                local_coord[d] = 0;
            }
        }

        global_max_abs = std::max(global_max_abs, blk_max_abs);
        global_max_rel = std::max(global_max_rel, blk_max_rel);
        global_compared += volume;
        global_passed = global_passed && blk_passed;
    }

    // Check for cuTENSOR blocks not in pipeline output (extra blocks)
    for (size_t i = 0; i < ref.block_coords.size(); ++i) {
        if (!ref_matched[i]) {
            // Check if the ref block is all zeros (within tolerance)
            bool all_zero = true;
            const T* ref_block = ref.block_data.data() + ref.block_offsets[i];
            for (int64_t e = 0; e < ref.block_volumes[i]; ++e) {
                if (static_cast<double>(std::abs(ref_block[e])) > atol) {
                    all_zero = false;
                    break;
                }
            }
            if (!all_zero) {
                std::cerr << "verify_against_block_sparse: cuTENSOR block [";
                for (size_t d = 0; d < ref.block_coords[i].size(); ++d) {
                    if (d) std::cerr << ",";
                    std::cerr << ref.block_coords[i][d];
                }
                std::cerr << "] not in pipeline output and is non-zero\n";
                global_passed = false;
            }
        }
    }

    vr.max_abs = global_max_abs;
    vr.max_rel = global_max_rel;
    vr.num_compared = global_compared;
    vr.passed = global_passed;
    return vr;
}

} // namespace bsptc
