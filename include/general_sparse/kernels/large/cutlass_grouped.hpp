#pragma once

// CRITICAL: CUTLASS headers MUST be included FIRST before any other headers
// to avoid CCCL (CUDA C++ Core Libraries) namespace conflicts with CUDA 13.x
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// Now include other headers
#include "../../work_partitioner.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <functional>

// CUTLASS Grouped GEMM execution for large block pairs.
// Routes "large" block contractions (M*N >= threshold) to optimized CUTLASS kernels.
namespace bsptc {

// Default scheduling mode for grouped GEMM
using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

// CUTLASS kernel configuration for single precision (float)
// Row-major layout for all matrices (as used in our block sparse format)
namespace cutlass_config {

// Single precision grouped GEMM kernel for SM80 (Ampere), NN (A=row, B=row)
using GemmKernelFloatNN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,                                          // ElementA
    cutlass::layout::RowMajor,                      // LayoutA
    cutlass::ComplexTransform::kNone,               // TransformA
    1,                                              // kAlignmentA (1 for row-major with arbitrary sizes)
    float,                                          // ElementB
    cutlass::layout::RowMajor,                      // LayoutB
    cutlass::ComplexTransform::kNone,               // TransformB
    1,                                              // kAlignmentB
    float,                                          // ElementC
    cutlass::layout::RowMajor,                      // LayoutC
    float,                                          // ElementAccumulator
    cutlass::arch::OpClassSimt,                     // OperatorClass (SIMT for flexibility)
    cutlass::arch::Sm80,                            // ArchTag
    cutlass::gemm::GemmShape<64, 64, 8>,            // ThreadblockShape
    cutlass::gemm::GemmShape<32, 32, 8>,            // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,              // InstructionShape (SIMT)
    cutlass::epilogue::thread::LinearCombination<
        float, 1, float, float>,                    // EpilogueOutputOp
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,                                              // Stages
    GroupScheduleMode::kDeviceOnly                  // Scheduling mode
>::GemmKernel;

// TN pass-through (A=col, B=row)
using GemmKernelFloatTN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// NT pass-through (A=row, B=col)
using GemmKernelFloatNT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// TT pass-through (A=col, B=col)
using GemmKernelFloatTT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmGroupedFloatNN = cutlass::gemm::device::GemmGrouped<GemmKernelFloatNN>;
using GemmGroupedFloatTN = cutlass::gemm::device::GemmGrouped<GemmKernelFloatTN>;
using GemmGroupedFloatNT = cutlass::gemm::device::GemmGrouped<GemmKernelFloatNT>;
using GemmGroupedFloatTT = cutlass::gemm::device::GemmGrouped<GemmKernelFloatTT>;

// Double precision grouped GEMM kernel for SM80 (Ampere), NN (A=row, B=row)
using GemmKernelDoubleNN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double,                                         // ElementA
    cutlass::layout::RowMajor,                      // LayoutA
    cutlass::ComplexTransform::kNone,               // TransformA
    1,                                              // kAlignmentA
    double,                                         // ElementB
    cutlass::layout::RowMajor,                      // LayoutB
    cutlass::ComplexTransform::kNone,               // TransformB
    1,                                              // kAlignmentB
    double,                                         // ElementC
    cutlass::layout::RowMajor,                      // LayoutC
    double,                                         // ElementAccumulator
    cutlass::arch::OpClassSimt,                     // OperatorClass
    cutlass::arch::Sm80,                            // ArchTag
    cutlass::gemm::GemmShape<32, 32, 8>,            // ThreadblockShape
    cutlass::gemm::GemmShape<16, 16, 8>,            // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,              // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        double, 1, double, double>,                 // EpilogueOutputOp
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,                                              // Stages
    GroupScheduleMode::kDeviceOnly                  // Scheduling mode
>::GemmKernel;

// TN pass-through (A=col, B=row)
using GemmKernelDoubleTN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    double,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    double,
    cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<16, 16, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// NT pass-through (A=row, B=col)
using GemmKernelDoubleNT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    double,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    double,
    cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<16, 16, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// TT pass-through (A=col, B=col)
using GemmKernelDoubleTT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    double,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    double,
    cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<16, 16, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// SM90 (Hopper) double-precision tensor core grouped GEMM kernels
// Uses f64 DMMA instruction (16x8x4) for ~20x over SIMT FP64
using GemmKernelDoubleSm90NN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmKernelDoubleSm90TN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmKernelDoubleSm90NT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmKernelDoubleSm90TT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmGroupedDoubleNN = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleNN>;
using GemmGroupedDoubleTN = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleTN>;
using GemmGroupedDoubleNT = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleNT>;
using GemmGroupedDoubleTT = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleTT>;

using GemmGroupedDoubleSm90NN = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleSm90NN>;
using GemmGroupedDoubleSm90TN = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleSm90TN>;
using GemmGroupedDoubleSm90NT = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleSm90NT>;
using GemmGroupedDoubleSm90TT = cutlass::gemm::device::GemmGrouped<GemmKernelDoubleSm90TT>;

} // namespace cutlass_config

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

// Prepare CUTLASS GEMM: allocate unified arg buffer + pinned staging, fill,
// async H2D, construct Gemm::Arguments.  Returns runner + buffers to free.
// No workspace (kDeviceOnly scheduling).
template <typename Gemm, typename Element>
inline CutlassGemmRunnerResult prepare_cutlass_gemm_impl(
    const LargeBlockWork<Element>& work,
    Element alpha,
    Element beta,
    cudaStream_t stream)
{
    CutlassGemmRunnerResult result;

    const int problem_count = static_cast<int>(work.num_gemm_problems);
    if (problem_count <= 0) return result;

    const size_t sizes[9] = {
        static_cast<size_t>(problem_count) * sizeof(cutlass::gemm::GemmCoord),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
    };
    const size_t d_total = total_allocation_size(sizes);
    const size_t h_total = d_total + sizes[0]; // staging + persistent h_problem_sizes

    // Allocate device and pinned host buffers
    check_cuda(cudaMalloc(&result.d_args_buf, d_total), "prepare_gemm: malloc args");
    check_cuda(cudaMallocHost(&result.h_args_buf, h_total), "prepare_gemm: malloc pinned");

    // Resolve device pointers
    void* d_ptrs[9];
    resolve_pointers(result.d_args_buf, sizes, d_ptrs);

    // Resolve host staging pointers (same layout)
    void* h_ptrs[9];
    resolve_pointers(result.h_args_buf, sizes, h_ptrs);

    // Persistent host problem_sizes lives after the staging region
    auto* h_problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(
        static_cast<char*>(result.h_args_buf) + d_total);

    // Fill staging buffer
    auto* h_staging_ps = static_cast<cutlass::gemm::GemmCoord*>(h_ptrs[0]);
    size_t prob_idx = 0;
    for (const auto& prob : work.problems) {
        for (int64_t b = 0; b < prob.batch_size; ++b) {
            auto coord = cutlass::gemm::GemmCoord(
                static_cast<int>(prob.M),
                static_cast<int>(prob.N),
                static_cast<int>(prob.K));
            h_staging_ps[prob_idx] = coord;
            h_problem_sizes[prob_idx] = coord;
            ++prob_idx;
        }
    }

    // Store patching locations
    result.d_ptr_A_loc = d_ptrs[1];
    result.d_ptr_B_loc = d_ptrs[2];
    result.h_ptr_A_loc = h_ptrs[1];
    result.h_ptr_B_loc = h_ptrs[2];
    result.ptr_array_bytes = sizes[1];  // sizes[1] == sizes[2]

    std::memcpy(h_ptrs[1], work.ptr_A.data(), sizes[1]);
    std::memcpy(h_ptrs[2], work.ptr_B.data(), sizes[2]);
    std::memcpy(h_ptrs[3], work.ptr_C.data(), sizes[3]);
    std::memcpy(h_ptrs[4], work.ptr_C.data(), sizes[4]); // D = C
    std::memcpy(h_ptrs[5], work.lda.data(), sizes[5]);
    std::memcpy(h_ptrs[6], work.ldb.data(), sizes[6]);
    std::memcpy(h_ptrs[7], work.ldc.data(), sizes[7]);
    std::memcpy(h_ptrs[8], work.ldd.data(), sizes[8]);

    // Single async H2D transfer
    check_cuda(cudaMemcpyAsync(result.d_args_buf, result.h_args_buf, d_total,
                                cudaMemcpyHostToDevice, stream),
               "prepare_gemm: memcpy unified");

    // Construct Gemm::Arguments
    int threadblock_count = Gemm::sufficient(h_problem_sizes, problem_count);
    if (threadblock_count == 0) {
        result.free_buffers();
        throw std::runtime_error("CUTLASS: insufficient resources for grouped GEMM");
    }

    typename Gemm::EpilogueOutputOp::Params epilogue_params(alpha, beta);

    typename Gemm::Arguments args(
        static_cast<cutlass::gemm::GemmCoord*>(d_ptrs[0]),
        problem_count,
        threadblock_count,
        epilogue_params,
        static_cast<Element**>(d_ptrs[1]),
        static_cast<Element**>(d_ptrs[2]),
        static_cast<Element**>(d_ptrs[3]),
        static_cast<Element**>(d_ptrs[4]),
        static_cast<int64_t*>(d_ptrs[5]),
        static_cast<int64_t*>(d_ptrs[6]),
        static_cast<int64_t*>(d_ptrs[7]),
        static_cast<int64_t*>(d_ptrs[8]),
        h_problem_sizes
    );

    // Runner: just initialize + run, no allocations (kDeviceOnly: workspace = nullptr)
    result.run = [args](cudaStream_t s) mutable {
        Gemm gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, s);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS grouped GEMM initialization failed");
        }
        // Clear any pre-existing CUDA error before launch — CUTLASS run() checks
        // cudaGetLastError() internally and would pick up stale errors from earlier
        // operations (e.g. cuTENSOR failures), causing spurious kErrorInternal.
        (void)cudaGetLastError();
        status = gemm_op.run(s);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS grouped GEMM execution failed");
        }
    };

    return result;
}

// Dispatch helper: select Gemm type based on work.gemm_mode, return runner.
template <typename T>
CutlassGemmRunnerResult prepare_cutlass_gemm_dispatch(
    const LargeBlockWork<T>& work,
    T alpha,
    T beta,
    cudaStream_t stream);

template <>
inline CutlassGemmRunnerResult prepare_cutlass_gemm_dispatch<float>(
    const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream)
{
    switch (work.gemm_mode) {
        case GemmMode::NN:
            return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedFloatNN>(work, alpha, beta, stream);
        case GemmMode::TN:
            return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedFloatTN>(work, alpha, beta, stream);
        case GemmMode::NT:
            return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedFloatNT>(work, alpha, beta, stream);
        case GemmMode::TT:
            return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedFloatTT>(work, alpha, beta, stream);
    }
    throw std::runtime_error("Unsupported CUTLASS GEMM mode (float)");
}

template <>
inline CutlassGemmRunnerResult prepare_cutlass_gemm_dispatch<double>(
    const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream)
{
    // Use f64 tensor cores (DMMA) on SM90+ (Hopper), fall back to SIMT on SM80
    if (get_device_sm_version() >= 90) {
        switch (work.gemm_mode) {
            case GemmMode::NN:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleSm90NN>(work, alpha, beta, stream);
            case GemmMode::TN:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleSm90TN>(work, alpha, beta, stream);
            case GemmMode::NT:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleSm90NT>(work, alpha, beta, stream);
            case GemmMode::TT:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleSm90TT>(work, alpha, beta, stream);
        }
    } else {
        switch (work.gemm_mode) {
            case GemmMode::NN:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleNN>(work, alpha, beta, stream);
            case GemmMode::TN:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleTN>(work, alpha, beta, stream);
            case GemmMode::NT:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleNT>(work, alpha, beta, stream);
            case GemmMode::TT:
                return prepare_cutlass_gemm_impl<cutlass_config::GemmGroupedDoubleTT>(work, alpha, beta, stream);
        }
    }
    throw std::runtime_error("Unsupported CUTLASS GEMM mode (double)");
}

// Check if CUTLASS grouped GEMM is available for the current GPU
inline bool is_cutlass_grouped_gemm_available() {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, 0);
    if (err != cudaSuccess) {
        return false;
    }
    // Require SM80 or higher (Ampere+)
    return props.major >= 8;
}

} // namespace bsptc
