#pragma once

#include "gpu_sparse_device.cuh"

#include <cstdint>
#include <vector>

// Contraction configuration, GEMM dimension helpers, and output volume computation.
// Hash-join pair matching lives in pair_matching.hpp (separate TU for gtl isolation).
namespace bsptc {

// Maximum supported tensor rank
constexpr int32_t MAX_RANK = 8;

// Contraction configuration passed to preprocessing kernel
struct ContractionConfig {
    // Positions of contracted dimensions in A and B
    int32_t contracted_pos_A[MAX_RANK];   // which dims in A are contracted
    int32_t contracted_pos_B[MAX_RANK];   // which dims in B are contracted
    dim_t num_contracted;                 // number of contracted dimensions

    // Positions of free dimensions in A and B (appear in output)
    int32_t free_pos_A[MAX_RANK];         // which dims in A are free
    int32_t free_pos_B[MAX_RANK];         // which dims in B are free
    dim_t num_free_A;
    dim_t num_free_B;

    // Positions of batch dimensions (appear in both inputs and output)
    int32_t batch_pos_A[MAX_RANK];        // which dims in A are batch
    int32_t batch_pos_B[MAX_RANK];        // which dims in B are batch
    dim_t num_batch;

    // Output dimension ordering (maps internal order to einsum output order)
    // internal_to_output[i] = position in einsum output for internal dim i
    // Internal order is: batch dims, then free_A, then free_B
    int32_t internal_to_output[MAX_RANK];
    dim_t output_rank;
};

// Host function: compute GEMM dimensions (M, N, K) for a block pair
// M = product of free_A dimensions
// N = product of free_B dimensions
// K = product of contracted dimensions
// batch_size = product of batch dimensions (optional, for batched GEMM)
inline void compute_gemm_dimensions(
    const dim_t* A_block_dims,
    const dim_t* B_block_dims,
    dim_t A_rank,
    dim_t B_rank,
    int64_t A_blk_idx,
    int64_t B_blk_idx,
    const ContractionConfig& config,
    dim_t& M,
    dim_t& N,
    dim_t& K,
    dim_t& batch_size)
{
    M = 1;
    N = 1;
    K = 1;
    batch_size = 1;

    // M = product of free_A dimensions
    for (int32_t i = 0; i < config.num_free_A; ++i) {
        int32_t pos_A = config.free_pos_A[i];
        M *= A_block_dims[A_blk_idx * A_rank + pos_A];
    }

    // N = product of free_B dimensions
    for (int32_t i = 0; i < config.num_free_B; ++i) {
        int32_t pos_B = config.free_pos_B[i];
        N *= B_block_dims[B_blk_idx * B_rank + pos_B];
    }

    // K = product of contracted dimensions
    for (int32_t i = 0; i < config.num_contracted; ++i) {
        int32_t pos_A = config.contracted_pos_A[i];
        K *= A_block_dims[A_blk_idx * A_rank + pos_A];
    }

    // batch_size = product of batch dimensions
    for (int32_t i = 0; i < config.num_batch; ++i) {
        int32_t pos_A = config.batch_pos_A[i];
        batch_size *= A_block_dims[A_blk_idx * A_rank + pos_A];
    }
}

// Overload that takes GPUSparseInput directly
template <typename T>
inline void compute_gemm_dimensions(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    int64_t A_blk_idx,
    int64_t B_blk_idx,
    const ContractionConfig& config,
    dim_t& M,
    dim_t& N,
    dim_t& K,
    dim_t& batch_size)
{
    compute_gemm_dimensions(
        A_gpu.block_dims.data(),
        B_gpu.block_dims.data(),
        A_gpu.rank,
        B_gpu.rank,
        A_blk_idx,
        B_blk_idx,
        config,
        M, N, K, batch_size);
}

// Host function: compute output block volume for a matched pair
inline int64_t compute_output_volume_host(
    const dim_t* A_block_dims,
    const dim_t* B_block_dims,
    dim_t A_rank,
    dim_t B_rank,
    int64_t A_blk_idx,
    int64_t B_blk_idx,
    const ContractionConfig& config)
{
    int64_t volume = 1;

    // Include batch dimensions
    for (int32_t i = 0; i < config.num_batch; ++i) {
        int32_t pos_A = config.batch_pos_A[i];
        volume *= A_block_dims[A_blk_idx * A_rank + pos_A];
    }

    // Include free dimensions from A
    for (int32_t i = 0; i < config.num_free_A; ++i) {
        int32_t pos_A = config.free_pos_A[i];
        volume *= A_block_dims[A_blk_idx * A_rank + pos_A];
    }

    // Include free dimensions from B
    for (int32_t i = 0; i < config.num_free_B; ++i) {
        int32_t pos_B = config.free_pos_B[i];
        volume *= B_block_dims[B_blk_idx * B_rank + pos_B];
    }

    return volume;
}

// Forward declaration — definition in pair_matching.hpp, compiled in
// pair_matching_instantiate.cu to avoid gtl/CUTLASS umul128 collision.
template <typename T>
std::vector<BlockPairInfo> find_matching_pairs(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config);

} // namespace bsptc
