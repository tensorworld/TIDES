#pragma once

#include "gemm_layout_check.hpp"
#include "warp_desc.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <vector>

// Shared descriptor structs consumed by all kernel tiers.
// Extracted from small_gemm_kernel.cuh, medium_gemm_kernel.cuh, work_partitioner.hpp.
namespace bsptc {

// ============================================================================
// Descriptor field groups (mixin bases for template composition)
// ============================================================================

struct F_Ptrs    { const void* A; const void* B; void* C; };
struct F_MNK     { dim_t M, N, K; };
struct F_Strides { dim_t lda, ldb, ldc; };

// Compose descriptors from field groups via multiple inheritance.
// All resulting types are trivially copyable POD.
template <typename... Fs>
struct Desc : Fs... {};

// ============================================================================
// Work partitioner structs
// ============================================================================

// Descriptor for a single GEMM problem derived from a block pair
struct GemmProblemDesc {
    int32_t pair_idx;       // Index into original BlockPairInfo array
    int32_t block_idx_A;    // Source block in A
    int32_t block_idx_B;    // Source block in B
    dim_t M, N, K;          // GEMM dimensions
    dim_t batch_size;       // Number of batch slices (1 if no batch dims)
    dim_t lda, ldb, ldc;    // Leading dimensions (row-major: lda=K, ldb=N, ldc=N)
    int64_t offset_A;       // Element offset into A values
    int64_t offset_B;       // Element offset into B values
    int64_t offset_C;       // Element offset into output buffer
};

// Collection of large block work for CUTLASS grouped GEMM
template <typename T>
struct LargeBlockWork {
    std::vector<GemmProblemDesc> problems;
    std::vector<size_t> source_indices;  // parallel to problems: index into source large.problems
    std::vector<T*> ptr_A;
    std::vector<T*> ptr_B;
    std::vector<T*> ptr_C;
    std::vector<int64_t> lda, ldb, ldc, ldd;
    int64_t total_output_elements = 0;
    int64_t num_gemm_problems = 0;
    GemmMode gemm_mode = GemmMode::NN;

    bool empty() const { return problems.empty(); }
    size_t size() const { return problems.size(); }
};

// Collection of block pairs for a non-CUTLASS tier (medium or small).
struct TierBlockWork {
    std::vector<int64_t> pair_indices;
    std::vector<GemmProblemDesc> problems;
    int64_t total_output_elements = 0;

    bool empty() const { return pair_indices.empty(); }
    size_t size() const { return pair_indices.size(); }
};

// Result of partitioning block pairs by threshold
template <typename T>
struct PartitionedWork {
    LargeBlockWork<T> large;
    TierBlockWork medium;
    TierBlockWork small;
    std::vector<BlockPairInfo> all_pairs;
};

// ============================================================================
// Utility: fill F_Ptrs + F_MNK + F_Strides fields on a Desc-based type.
// ============================================================================

template <typename DescT, typename T>
DescT make_stride_desc(const T* A, const T* B, T* C,
                       dim_t M, dim_t N, dim_t K,
                       dim_t lda, dim_t ldb, dim_t ldc) {
    static_assert(std::is_base_of_v<F_Ptrs, DescT> && std::is_base_of_v<F_Strides, DescT>);
    DescT sd{};
    sd.A = A; sd.B = B; sd.C = C;
    sd.M = M; sd.N = N; sd.K = K;
    sd.lda = lda; sd.ldb = ldb; sd.ldc = ldc;
    return sd;
}

// ============================================================================
// Warp-level GEMM types (defined in warp_desc.hpp, included at top)
// ============================================================================

static_assert(std::is_trivially_copyable_v<WarpShape<StdDim, StdK>>);
static_assert(std::is_trivially_copyable_v<WarpShape<IntDim, StdK>>);
static_assert(std::is_trivially_copyable_v<WarpShape<StdDim, IntK>>);
static_assert(std::is_trivially_copyable_v<WarpShape<IntDim, IntK>>);
static_assert(std::is_trivially_copyable_v<WarpMem>);

} // namespace bsptc
