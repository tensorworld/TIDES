#pragma once

// Lightweight forward-declaration header for CUTLASS grouped GEMM.
// Does NOT include any CUTLASS headers — safe to include from pipeline.cuh
// without incurring heavy template instantiation costs on every rebuild.
//
// The actual CUTLASS kernel types and implementations live in
// src/cutlass/gemm_*.cu (per-shape files, compiled in parallel).

#include "../../pipeline_descs.hpp"
#include "large_backend.hpp"
#include "../../gpu_sparse_device.cuh"
#include "../../cuda_buffer_pool.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <utility>

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

// Result of prepare_cutlass_gemm: a callable runner + pooled buffers to
// release after execution (free_buffers returns them to the process-wide
// pools in cuda_buffer_pool.hpp for reuse by later pipeline invocations).
struct CutlassGemmRunnerResult {
    std::function<void(cudaStream_t)> run;
    void* d_args_buf = nullptr;   // from device_args_pool(); release via free_buffers()
    void* h_args_buf = nullptr;   // from pinned_args_pool(); release via free_buffers()

    // For patching A/B pointers after transpose (locations within the buffers)
    void* d_ptr_A_loc = nullptr;  // device array of A pointers
    void* d_ptr_B_loc = nullptr;  // device array of B pointers
    void* h_ptr_A_loc = nullptr;  // pinned host staging for A pointers
    void* h_ptr_B_loc = nullptr;  // pinned host staging for B pointers
    size_t ptr_array_bytes = 0;   // bytes per pointer array

    // Move-only: copying would double-release the pooled buffers (the second
    // release cudaFrees a pointer still on the pool's free list).
    CutlassGemmRunnerResult() = default;
    CutlassGemmRunnerResult(const CutlassGemmRunnerResult&) = delete;
    CutlassGemmRunnerResult& operator=(const CutlassGemmRunnerResult&) = delete;
    CutlassGemmRunnerResult(CutlassGemmRunnerResult&& o) noexcept { take_from(o); }
    CutlassGemmRunnerResult& operator=(CutlassGemmRunnerResult&& o) noexcept {
        if (this != &o) { free_buffers(); take_from(o); }
        return *this;
    }
    ~CutlassGemmRunnerResult() { free_buffers(); }

    // Take o's fields and leave o empty (null run, null pointers), so the
    // moved-from object's destructor, free_buffers(), and patch_ab_pointers()
    // are all no-ops. A swap here would hand the destination's stale closure
    // and patch pointers to the source.
    void take_from(CutlassGemmRunnerResult& o) noexcept {
        run = std::move(o.run);
        o.run = nullptr;
        d_args_buf = std::exchange(o.d_args_buf, nullptr);
        h_args_buf = std::exchange(o.h_args_buf, nullptr);
        d_ptr_A_loc = std::exchange(o.d_ptr_A_loc, nullptr);
        d_ptr_B_loc = std::exchange(o.d_ptr_B_loc, nullptr);
        h_ptr_A_loc = std::exchange(o.h_ptr_A_loc, nullptr);
        h_ptr_B_loc = std::exchange(o.h_ptr_B_loc, nullptr);
        ptr_array_bytes = std::exchange(o.ptr_array_bytes, size_t{0});
    }

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

    // Return buffers to the pools. Caller must guarantee no in-flight device
    // work still uses them (the pipeline's sync points ensure this; the old
    // cudaFreeHost implicitly synchronized, pool release does not).
    void free_buffers() {
        if (d_args_buf) { device_args_pool().release(d_args_buf); d_args_buf = nullptr; }
        if (h_args_buf) { pinned_args_pool().release(h_args_buf); h_args_buf = nullptr; }
    }
};

// Defined in src/cublas/cublas_grouped_gemm.cu.  Declared here (rather than
// included) so the shared arg buffers can be sized for whichever Large-tier
// backend the per-queue selector picks, without a circular include.
size_t cublas_args_device_bytes(int count);
size_t cublas_args_host_bytes(int count);

// Allocate Large-tier arg buffers sized for max_problem_count.
// The streaming arg buffers are allocated once and reused across shape queues,
// and different queues may resolve to different backends, so each buffer is
// sized for the larger of the two layouts.
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
    size_t d_total = total_allocation_size(sizes);
    size_t h_total = d_total + sizes[0];

    // Fit either backend's region layout in the same pooled buffers.
    const size_t cublas_d = cublas_args_device_bytes(max_problem_count);
    const size_t cublas_h = cublas_args_host_bytes(max_problem_count);
    if (cublas_d > d_total) d_total = cublas_d;
    if (cublas_h > h_total) h_total = cublas_h;

    result.d_args_buf = device_args_pool().acquire(d_total);
    result.h_args_buf = pinned_args_pool().acquire(h_total);
    result.ptr_array_bytes = sizes[1];
    return result;
}

// Forward declarations — implementations in src/cutlass/gemm_dispatch.cu

// Tile-only entry points (CUTLASS backend).
template <typename T>
CutlassGemmRunnerResult prepare_cutlass_tile_dispatch(
    const LargeBlockWork<T>& work,
    T alpha, T beta,
    TileShape tile, cudaStream_t stream);

template <typename T>
void fill_cutlass_tile_dispatch(
    CutlassGemmRunnerResult& result,
    const LargeBlockWork<T>& work,
    T alpha, T beta,
    TileShape tile, cudaStream_t stream);

// Backend-aware entry points — what the pipeline calls.  Routes to CUTLASS or
// cuBLAS according to the KernelChoice produced by best_kernel_for_queue.
template <typename T>
CutlassGemmRunnerResult prepare_large_gemm_dispatch(
    const LargeBlockWork<T>& work,
    T alpha, T beta,
    KernelChoice choice, cudaStream_t stream);

template <typename T>
void fill_large_args_dispatch(
    CutlassGemmRunnerResult& result,
    const LargeBlockWork<T>& work,
    T alpha, T beta,
    KernelChoice choice, cudaStream_t stream);

bool is_cutlass_grouped_gemm_available();

} // namespace bsptc
