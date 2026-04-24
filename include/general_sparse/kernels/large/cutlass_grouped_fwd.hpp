#pragma once

// Lightweight forward-declaration header for CUTLASS grouped GEMM.
// Does NOT include any CUTLASS headers — safe to include from pipeline.cuh
// without incurring heavy template instantiation costs on every rebuild.
//
// The actual CUTLASS kernel types and implementations live in
// src/cutlass/gemm_*.cu (per-shape files, compiled in parallel).

#include "../../pipeline_descs.hpp"
#include "../../gpu_sparse_device.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <cstring>

namespace bsptc {

constexpr size_t GEMM_ARG_ALIGN = 128;

// Alignment helper: round up to alignment boundary
inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Compute total allocation size for N regions with alignment padding
template <size_t N>
size_t total_allocation_size(const size_t (&sizes)[N], size_t alignment = GEMM_ARG_ALIGN) {
    size_t total = 0;
    for (size_t i = 0; i < N; ++i) {
        total = align_up(total, alignment) + sizes[i];
    }
    return total;
}

// Resolve N pointers into a base buffer given region sizes
template <size_t N>
void resolve_pointers(void* base, const size_t (&sizes)[N], void* (&ptrs)[N],
                      size_t alignment = GEMM_ARG_ALIGN) {
    char* p = static_cast<char*>(base);
    for (size_t i = 0; i < N; ++i) {
        p = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(p), alignment));
        ptrs[i] = p;
        p += sizes[i];
    }
}

// Result of prepare_cutlass_gemm: a callable runner + buffers to free after execution.
struct CutlassGemmRunnerResult {
    std::function<void(cudaStream_t)> run;
    void* d_args_buf = nullptr;   // caller cudaFree after run()
    void* h_args_buf = nullptr;   // caller cudaFreeHost after run()

    // For patching A/B pointers after transpose (locations within the buffers)
    void* d_ptr_A_loc = nullptr;  // device array of A pointers
    void* d_ptr_B_loc = nullptr;  // device array of B pointers
    void* h_ptr_A_loc = nullptr;  // pinned host staging for A pointers
    void* h_ptr_B_loc = nullptr;  // pinned host staging for B pointers
    size_t ptr_array_bytes = 0;   // bytes per pointer array

    explicit operator bool() const { return static_cast<bool>(run); }

    // Patch A/B pointer arrays on device after transpose changes base pointers.
    // Caller rebuilds h_ptr_A_loc / h_ptr_B_loc, then calls this.
    void patch_ab_pointers(cudaStream_t stream) {
        if (ptr_array_bytes > 0 && d_ptr_A_loc && h_ptr_A_loc) {
            check_cuda(cudaMemcpyAsync(d_ptr_A_loc, h_ptr_A_loc, ptr_array_bytes,
                                        cudaMemcpyHostToDevice, stream),
                       "patch ptr_A");
        }
        if (ptr_array_bytes > 0 && d_ptr_B_loc && h_ptr_B_loc) {
            check_cuda(cudaMemcpyAsync(d_ptr_B_loc, h_ptr_B_loc, ptr_array_bytes,
                                        cudaMemcpyHostToDevice, stream),
                       "patch ptr_B");
        }
    }

    void free_buffers() {
        if (d_args_buf) { cudaFree(d_args_buf); d_args_buf = nullptr; }
        if (h_args_buf) { cudaFreeHost(h_args_buf); h_args_buf = nullptr; }
    }
};

// Allocate CUTLASS arg buffers sized for max_problem_count (no CUTLASS headers needed).
inline CutlassGemmRunnerResult allocate_cutlass_args(int max_problem_count) {
    CutlassGemmRunnerResult result;
    if (max_problem_count <= 0) return result;

    // 9 arrays: GemmCoord(12B) + 4 ptrs(8B) + 4 strides(8B) per problem
    // Use sizeof(void*) and sizeof(int64_t) to avoid CUTLASS dependency.
    const size_t sizes[9] = {
        static_cast<size_t>(max_problem_count) * 12,  // GemmCoord (3 x int32)
        static_cast<size_t>(max_problem_count) * sizeof(void*),
        static_cast<size_t>(max_problem_count) * sizeof(void*),
        static_cast<size_t>(max_problem_count) * sizeof(void*),
        static_cast<size_t>(max_problem_count) * sizeof(void*),
        static_cast<size_t>(max_problem_count) * sizeof(int64_t),
        static_cast<size_t>(max_problem_count) * sizeof(int64_t),
        static_cast<size_t>(max_problem_count) * sizeof(int64_t),
        static_cast<size_t>(max_problem_count) * sizeof(int64_t),
    };
    const size_t d_total = total_allocation_size(sizes);
    const size_t h_total = d_total + sizes[0];

    check_cuda(cudaMalloc(&result.d_args_buf, d_total), "alloc_args: malloc device");
    check_cuda(cudaMallocHost(&result.h_args_buf, h_total), "alloc_args: malloc pinned");
    result.ptr_array_bytes = sizes[1];
    return result;
}

// Tile shape enum for multi-tile CUTLASS dispatch
enum class TileShape;

// Forward declarations — implementations in src/cutlass/gemm_dispatch.cu

template <typename T>
CutlassGemmRunnerResult prepare_cutlass_gemm_dispatch(
    const LargeBlockWork<T>& work,
    T alpha, T beta,
    TileShape tile, cudaStream_t stream);

template <typename T>
void fill_cutlass_args_dispatch(
    CutlassGemmRunnerResult& result,
    const LargeBlockWork<T>& work,
    T alpha, T beta,
    TileShape tile, cudaStream_t stream);

bool is_cutlass_grouped_gemm_available();

} // namespace bsptc
