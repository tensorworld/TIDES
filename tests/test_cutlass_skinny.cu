// Correctness test + benchmark for 8x64 CUTLASS skinny tile shapes.
//
// Three kernel variants tested:
//   1. RowMajor-C (direct): works for both skinny-wide and skinny-tall via predication
//   2. Fused transpose (ColumnMajor-C + TransposePermute PermuteDLayout):
//      kInternalTranspose swaps M/N internally so the 8x64 tile sees [N x M],
//      then TransposePermute writes row-major [M x N] in the epilogue.
//      Same output layout as direct RowC — no separate transpose kernel needed.
//
// Key insight: do NOT manually swap M/N in GemmCoord. Pass original (M, N, K).
// kInternalTranspose (triggered by ColumnMajor-C) handles the swap internally.

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/permute.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <type_traits>

// ============================================================================
// TransposePermute: fused epilogue transpose via CUTLASS PermuteDLayout.
//
// With LayoutC = ColumnMajor, CUTLASS sets kInternalTranspose = true:
//   - Swaps M/N internally: M_int = N_ext, N_int = M_ext
//   - Swaps A <-> B pointers/strides and transposes their layouts
//   - Internal LayoutC becomes RowMajor (via MapArguments)
//
// The epilogue's PredicatedTileIterator iterates over internal RowMajor C:
//   - col iterates over N_int (= M_ext) — the original M coordinate
//   - row iterates over M_int (= N_ext) — the original N coordinate
//   - Calls: permute_layout_(PitchLinearCoord(col, row))
//
// Constructor receives PitchLinearCoord(extent.column(), extent.row())
// where extent = problem_size.mn() = MatrixCoord(M_int, N_int):
//   - contiguous = N_int = M_ext (large dim)
//   - strided = M_int = N_ext (small dim)
//
// For row-major output C[M_ext x N_ext]:
//   offset = col * N_ext + row = m * N + n
// ============================================================================

class TransposePermute : public cutlass::layout::PermuteBase {
private:
    Index N_orig_;
public:
    CUTLASS_HOST_DEVICE
    TransposePermute(cutlass::layout::PitchLinearCoord extent, Index stride) {
        N_orig_ = extent.strided();  // M_int = N_ext
        (void)stride;
    }

    // coord.contiguous() = col = M_coord, coord.strided() = row = N_coord
    CUTLASS_HOST_DEVICE
    LongIndex operator()(cutlass::layout::PitchLinearCoord coord) const {
        return LongIndex(coord.contiguous()) * LongIndex(N_orig_) + LongIndex(coord.strided());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(cutlass::MatrixCoord coord) const {
        return operator()(cutlass::layout::PitchLinearCoord(coord.column(), coord.row()));
    }
};

// ============================================================================
// Kernel type definitions
// ============================================================================

using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

// 8x64 RowMajor-C — works for both skinny-wide (M small) and skinny-tall (N small)
template <typename LayoutA, typename LayoutB>
using GemmKernel_8x64_RowC = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<8, 64, 8>,  // 1 warp, 32 threads
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// Multi-warp RowC variants for benchmarking
// 2 warps (64 threads): threadblock <8,64,8>, warp <8,32,8>
template <typename LayoutA, typename LayoutB>
using GemmKernel_8x64_RowC_2w = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<8, 32, 8>,  // 2 warps in N
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// 4 warps (128 threads): threadblock <8,128,8>, warp <8,32,8>
template <typename LayoutA, typename LayoutB>
using GemmKernel_8x128_RowC_4w = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 128, 8>,
    cutlass::gemm::GemmShape<8, 32, 8>,  // 4 warps in N
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// Fused transpose variants (ColumnMajor-C + TransposePermute)
// 1 warp (32 threads)
template <typename LayoutA, typename LayoutB>
using GemmKernel_8x64_Fused_1w = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::ColumnMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
        float, float, float, float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    TransposePermute
>::GemmKernel;

// 2 warps (64 threads)
template <typename LayoutA, typename LayoutB>
using GemmKernel_8x64_Fused_2w = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::ColumnMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<8, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
        float, float, float, float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    TransposePermute
>::GemmKernel;

// 4 warps (128 threads)
template <typename LayoutA, typename LayoutB>
using GemmKernel_8x128_Fused_4w = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::ColumnMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 128, 8>,
    cutlass::gemm::GemmShape<8, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
        float, float, float, float>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    TransposePermute
>::GemmKernel;

// ============================================================================
// Helpers
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void cpu_gemm(int M, int N, int K,
              float alpha, const float* A, int lda, bool transA,
              const float* B, int ldb, bool transB,
              float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + i] : A[i * lda + k];
                float b_val = transB ? B[j * ldb + k] : B[k * ldb + j];
                sum += a_val * b_val;
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

void fill_random(float* data, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        data[i] = 2.0f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 1.0f;
    }
}

float compare(const float* ref, const float* gpu, int M, int N, int ldc, float tol) {
    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float r = ref[i * ldc + j];
            float g = gpu[i * ldc + j];
            float err = std::fabs(r - g);
            float scale = std::max(1.0f, std::max(std::fabs(r), std::fabs(g)));
            float rel = err / scale;
            if (rel > max_err) max_err = rel;
            if (rel > tol && err_count < 5) {
                printf("  MISMATCH at (%d,%d): ref=%.6f gpu=%.6f err=%.6f\n", i, j, r, g, err);
                err_count++;
            }
        }
    }
    return max_err;
}

// ============================================================================
// Grouped GEMM runners (single problem)
// ============================================================================

template <typename GemmDevice>
bool run_grouped_gemm(int M, int N, int K,
                      float* d_A, int lda,
                      float* d_B, int ldb,
                      float* d_C, int ldc,
                      float alpha, float beta,
                      cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    cutlass::gemm::GemmCoord* d_problem_sizes;
    float** d_ptr_A; float** d_ptr_B; float** d_ptr_C; float** d_ptr_D;
    int64_t* d_lda; int64_t* d_ldb; int64_t* d_ldc; int64_t* d_ldd;

    CHECK_CUDA(cudaMalloc(&d_problem_sizes, sizeof(cutlass::gemm::GemmCoord)));
    CHECK_CUDA(cudaMalloc(&d_ptr_A, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_ptr_B, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_ptr_C, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_ptr_D, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_lda, sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_ldb, sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_ldc, sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_ldd, sizeof(int64_t)));

    CHECK_CUDA(cudaMemcpy(d_problem_sizes, &problem_size, sizeof(problem_size), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_A, &d_A, sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_B, &d_B, sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_C, &d_C, sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_D, &d_C, sizeof(float*), cudaMemcpyHostToDevice));
    int64_t h_lda = lda, h_ldb = ldb, h_ldc = ldc;
    CHECK_CUDA(cudaMemcpy(d_lda, &h_lda, sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldb, &h_ldb, sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldc, &h_ldc, sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldd, &h_ldc, sizeof(int64_t), cudaMemcpyHostToDevice));

    cutlass::gemm::GemmCoord h_problem_sizes = problem_size;

    int threadblock_count = GemmDevice::sufficient(&h_problem_sizes, 1);
    if (threadblock_count == 0) {
        printf("  insufficient resources\n");
        cudaFree(d_problem_sizes); cudaFree(d_ptr_A); cudaFree(d_ptr_B);
        cudaFree(d_ptr_C); cudaFree(d_ptr_D);
        cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc); cudaFree(d_ldd);
        return false;
    }

    typename GemmDevice::EpilogueOutputOp::Params epilogue_params(alpha, beta);

    typename GemmDevice::Arguments args(
        d_problem_sizes, 1, threadblock_count, epilogue_params,
        d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_D,
        d_lda, d_ldb, d_ldc, d_ldd,
        &h_problem_sizes
    );

    GemmDevice gemm_op;
    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        printf("  init failed: %d\n", (int)status);
        cudaFree(d_problem_sizes); cudaFree(d_ptr_A); cudaFree(d_ptr_B);
        cudaFree(d_ptr_C); cudaFree(d_ptr_D);
        cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc); cudaFree(d_ldd);
        return false;
    }

    (void)cudaGetLastError();
    status = gemm_op.run(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaFree(d_problem_sizes); cudaFree(d_ptr_A); cudaFree(d_ptr_B);
    cudaFree(d_ptr_C); cudaFree(d_ptr_D);
    cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc); cudaFree(d_ldd);

    if (status != cutlass::Status::kSuccess) {
        printf("  run failed: %d\n", (int)status);
        return false;
    }
    return true;
}

// Fused transpose runner: pass original (M, N, K), set ldc = M for internal RowMajor.
// kInternalTranspose swaps M/N and A/B internally.
// TransposePermute writes row-major [M x N] output directly.
template <typename GemmDevice>
bool run_grouped_gemm_fused(int M, int N, int K,
                             float* d_A, int lda,
                             float* d_B, int ldb,
                             float* d_C,
                             float alpha, float beta,
                             cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    cutlass::gemm::GemmCoord* d_problem_sizes;
    float** d_ptr_A; float** d_ptr_B; float** d_ptr_C; float** d_ptr_D;
    int64_t* d_lda; int64_t* d_ldb; int64_t* d_ldc; int64_t* d_ldd;

    CHECK_CUDA(cudaMalloc(&d_problem_sizes, sizeof(cutlass::gemm::GemmCoord)));
    CHECK_CUDA(cudaMalloc(&d_ptr_A, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_ptr_B, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_ptr_C, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_ptr_D, sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_lda, sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_ldb, sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_ldc, sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_ldd, sizeof(int64_t)));

    CHECK_CUDA(cudaMemcpy(d_problem_sizes, &problem_size, sizeof(problem_size), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_A, &d_A, sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_B, &d_B, sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_C, &d_C, sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptr_D, &d_C, sizeof(float*), cudaMemcpyHostToDevice));
    int64_t h_lda = lda, h_ldb = ldb;
    // ldc = M: internal RowMajor stride = N_int = M_ext.
    // With PermuteD=true, only affects predication, not store addresses.
    int64_t h_ldc = M;
    CHECK_CUDA(cudaMemcpy(d_lda, &h_lda, sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldb, &h_ldb, sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldc, &h_ldc, sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldd, &h_ldc, sizeof(int64_t), cudaMemcpyHostToDevice));

    cutlass::gemm::GemmCoord h_problem_sizes = problem_size;

    int threadblock_count = GemmDevice::sufficient(&h_problem_sizes, 1);
    if (threadblock_count == 0) {
        printf("  insufficient resources\n");
        cudaFree(d_problem_sizes); cudaFree(d_ptr_A); cudaFree(d_ptr_B);
        cudaFree(d_ptr_C); cudaFree(d_ptr_D);
        cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc); cudaFree(d_ldd);
        return false;
    }

    typename GemmDevice::EpilogueOutputOp::Params epilogue_params(alpha, beta);

    typename GemmDevice::Arguments args(
        d_problem_sizes, 1, threadblock_count, epilogue_params,
        d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_D,
        d_lda, d_ldb, d_ldc, d_ldd,
        &h_problem_sizes
    );

    GemmDevice gemm_op;
    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        printf("  init failed: %d\n", (int)status);
        cudaFree(d_problem_sizes); cudaFree(d_ptr_A); cudaFree(d_ptr_B);
        cudaFree(d_ptr_C); cudaFree(d_ptr_D);
        cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc); cudaFree(d_ldd);
        return false;
    }

    (void)cudaGetLastError();
    status = gemm_op.run(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaFree(d_problem_sizes); cudaFree(d_ptr_A); cudaFree(d_ptr_B);
    cudaFree(d_ptr_C); cudaFree(d_ptr_D);
    cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc); cudaFree(d_ldd);

    if (status != cutlass::Status::kSuccess) {
        printf("  run failed: %d\n", (int)status);
        return false;
    }
    return true;
}

// ============================================================================
// Test harness
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    float max_err;
};

using GemmNN_RowC = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_RowC<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmTN_RowC = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_RowC<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>>;
using GemmNT_RowC = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_RowC<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>>;
using GemmTT_RowC = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_RowC<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>>;

using GemmNN_RowC_2w = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_RowC_2w<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmNN_RowC_4w = cutlass::gemm::device::GemmGrouped<GemmKernel_8x128_RowC_4w<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

using GemmNN_Fused = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_Fused_1w<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmTN_Fused = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_Fused_1w<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>>;
using GemmNT_Fused = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_Fused_1w<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>>;
using GemmTT_Fused = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_Fused_1w<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>>;

using GemmNN_Fused_2w = cutlass::gemm::device::GemmGrouped<GemmKernel_8x64_Fused_2w<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmNN_Fused_4w = cutlass::gemm::device::GemmGrouped<GemmKernel_8x128_Fused_4w<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

// Test: direct RowMajor-C kernel (works for any M, N via predication)
template <typename GemmDevice>
TestResult test_direct(const char* tag, const char* mode_name, int M, int N, int K,
                       bool transA, bool transB, unsigned seed) {
    std::string name = std::string(tag) + "_" + mode_name +
                       "_M" + std::to_string(M) + "_N" + std::to_string(N) + "_K" + std::to_string(K);

    int A_rows = transA ? K : M, A_cols = transA ? M : K;
    int B_rows = transB ? N : K, B_cols = transB ? K : N;
    int lda = A_cols, ldb = B_cols, ldc = N;

    std::vector<float> h_A(A_rows * A_cols), h_B(B_rows * B_cols), h_C(M * N, 0.0f);
    fill_random(h_A.data(), h_A.size(), seed);
    fill_random(h_B.data(), h_B.size(), seed + 1);

    std::vector<float> ref_C(M * N, 0.0f);
    cpu_gemm(M, N, K, 1.0f, h_A.data(), lda, transA, h_B.data(), ldb, transB, 0.0f, ref_C.data(), ldc);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, h_C.size() * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    bool ok = run_grouped_gemm<GemmDevice>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, 1.0f, 0.0f, stream);

    TestResult result{name, false, 999.0f};
    if (ok) {
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float tol = 1e-4f;
        result.max_err = compare(ref_C.data(), h_C.data(), M, N, ldc, tol);
        result.passed = (result.max_err < tol);
    } else {
        printf("  (kernel failed)\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaStreamDestroy(stream);
    return result;
}

// Test: fused transpose kernel — output is row-major [M x N] with ldc = N
template <typename GemmDevice>
TestResult test_fused(const char* mode_name, int M, int N, int K,
                      bool transA, bool transB, unsigned seed) {
    std::string name = std::string("Fused_") + mode_name +
                       "_M" + std::to_string(M) + "_N" + std::to_string(N) + "_K" + std::to_string(K);

    int A_rows = transA ? K : M, A_cols = transA ? M : K;
    int B_rows = transB ? N : K, B_cols = transB ? K : N;
    int lda = A_cols, ldb = B_cols, ldc = N;

    std::vector<float> h_A(A_rows * A_cols), h_B(B_rows * B_cols), h_C(M * N, 0.0f);
    fill_random(h_A.data(), h_A.size(), seed);
    fill_random(h_B.data(), h_B.size(), seed + 1);

    std::vector<float> ref_C(M * N, 0.0f);
    cpu_gemm(M, N, K, 1.0f, h_A.data(), lda, transA, h_B.data(), ldb, transB, 0.0f, ref_C.data(), ldc);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, h_C.size() * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    bool ok = run_grouped_gemm_fused<GemmDevice>(
        M, N, K, d_A, lda, d_B, ldb, d_C, 1.0f, 0.0f, stream);

    TestResult result{name, false, 999.0f};
    if (ok) {
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float tol = 1e-4f;
        result.max_err = compare(ref_C.data(), h_C.data(), M, N, ldc, tol);
        result.passed = (result.max_err < tol);
    } else {
        printf("  (kernel failed)\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaStreamDestroy(stream);
    return result;
}

// Test: fused transpose kernel with beta=1 accumulation
// Two GEMM passes: first with beta=0, then with beta=1 on same output buffer.
// Verifies that the patched load path reads from TransposePermute-layout addresses.
template <typename GemmDevice>
TestResult test_fused_beta1(const char* mode_name, int M, int N, int K,
                            bool transA, bool transB, unsigned seed) {
    std::string name = std::string("FusedBeta1_") + mode_name +
                       "_M" + std::to_string(M) + "_N" + std::to_string(N) + "_K" + std::to_string(K);

    int A_rows = transA ? K : M, A_cols = transA ? M : K;
    int B_rows = transB ? N : K, B_cols = transB ? K : N;
    int lda = A_cols, ldb = B_cols, ldc = N;

    // Two different input matrices for two passes
    std::vector<float> h_A1(A_rows * A_cols), h_B1(B_rows * B_cols);
    std::vector<float> h_A2(A_rows * A_cols), h_B2(B_rows * B_cols);
    fill_random(h_A1.data(), h_A1.size(), seed);
    fill_random(h_B1.data(), h_B1.size(), seed + 1);
    fill_random(h_A2.data(), h_A2.size(), seed + 100);
    fill_random(h_B2.data(), h_B2.size(), seed + 101);

    // CPU reference: C = A1*B1, then C += A2*B2
    std::vector<float> ref_C(M * N, 0.0f);
    cpu_gemm(M, N, K, 1.0f, h_A1.data(), lda, transA, h_B1.data(), ldb, transB, 0.0f, ref_C.data(), ldc);
    cpu_gemm(M, N, K, 1.0f, h_A2.data(), lda, transA, h_B2.data(), ldb, transB, 1.0f, ref_C.data(), ldc);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Pass 1: beta=0
    CHECK_CUDA(cudaMemcpy(d_A, h_A1.data(), h_A1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B1.data(), h_B1.size() * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = run_grouped_gemm_fused<GemmDevice>(M, N, K, d_A, lda, d_B, ldb, d_C, 1.0f, 0.0f, stream);

    TestResult result{name, false, 999.0f};
    if (!ok) { printf("  (pass 1 failed)\n"); goto cleanup; }

    // Pass 2: beta=1 (accumulate onto existing row-major output)
    CHECK_CUDA(cudaMemcpy(d_A, h_A2.data(), h_A2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B2.data(), h_B2.size() * sizeof(float), cudaMemcpyHostToDevice));
    ok = run_grouped_gemm_fused<GemmDevice>(M, N, K, d_A, lda, d_B, ldb, d_C, 1.0f, 1.0f, stream);

    if (ok) {
        std::vector<float> h_C(M * N);
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float tol = 1e-4f;
        result.max_err = compare(ref_C.data(), h_C.data(), M, N, ldc, tol);
        result.passed = (result.max_err < tol);
    } else {
        printf("  (pass 2 failed)\n");
    }

cleanup:
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaStreamDestroy(stream);
    return result;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== CUTLASS 8x64 Skinny Tile Correctness Tests ===\n\n");

    std::vector<TestResult> results;

    // ----- Test 1: Skinny-wide with RowMajor-C (M small, N large) -----
    printf("--- Test 1: Skinny-wide RowC (M=8, N=200) ---\n");
    results.push_back(test_direct<GemmNN_RowC>("Wide_RowC", "NN", 8, 200, 16, false, false, 42));
    results.push_back(test_direct<GemmTN_RowC>("Wide_RowC", "TN", 8, 200, 16, true,  false, 43));
    results.push_back(test_direct<GemmNT_RowC>("Wide_RowC", "NT", 8, 200, 16, false, true,  44));
    results.push_back(test_direct<GemmTT_RowC>("Wide_RowC", "TT", 8, 200, 16, true,  true,  45));

    printf("\n--- Test 1b: Skinny-wide RowC (M=8, N=10000) ---\n");
    results.push_back(test_direct<GemmNN_RowC>("Wide_RowC", "NN", 8, 10000, 32, false, false, 100));

    printf("\n--- Test 1c: Skinny-wide RowC (M=5, N=200) ---\n");
    results.push_back(test_direct<GemmNN_RowC>("Wide_RowC", "NN", 5, 200, 16, false, false, 200));

    // ----- Test 2: Skinny-tall with direct RowC (predication, no transpose) -----
    printf("\n--- Test 2: Skinny-tall Direct RowC (M=200, N=8) ---\n");
    results.push_back(test_direct<GemmNN_RowC>("Tall_RowC", "NN", 200, 8, 16, false, false, 42));
    results.push_back(test_direct<GemmTN_RowC>("Tall_RowC", "TN", 200, 8, 16, true,  false, 43));
    results.push_back(test_direct<GemmNT_RowC>("Tall_RowC", "NT", 200, 8, 16, false, true,  44));
    results.push_back(test_direct<GemmTT_RowC>("Tall_RowC", "TT", 200, 8, 16, true,  true,  45));

    printf("\n--- Test 2b: Skinny-tall Direct RowC (M=10000, N=8) ---\n");
    results.push_back(test_direct<GemmNN_RowC>("Tall_RowC", "NN", 10000, 8, 32, false, false, 300));

    printf("\n--- Test 2c: Skinny-tall Direct RowC edge cases ---\n");
    results.push_back(test_direct<GemmNN_RowC>("Tall_RowC", "NN", 200, 5, 16, false, false, 400));
    results.push_back(test_direct<GemmNN_RowC>("Tall_RowC", "NN", 200, 1, 16, false, false, 500));

    // ----- Test 3: Fused transpose (ColumnMajor-C + TransposePermute) -----
    // Output is identical row-major [M x N] as direct RowC
    printf("\n--- Test 3: Fused transpose (M=200, N=8) ---\n");
    results.push_back(test_fused<GemmNN_Fused>("NN", 200, 8, 16, false, false, 42));
    results.push_back(test_fused<GemmTN_Fused>("TN", 200, 8, 16, true,  false, 43));
    results.push_back(test_fused<GemmNT_Fused>("NT", 200, 8, 16, false, true,  44));
    results.push_back(test_fused<GemmTT_Fused>("TT", 200, 8, 16, true,  true,  45));

    printf("\n--- Test 3b: Fused transpose (M=10000, N=8) ---\n");
    results.push_back(test_fused<GemmNN_Fused>("NN", 10000, 8, 32, false, false, 300));
    results.push_back(test_fused<GemmTN_Fused>("TN", 10000, 8, 32, true,  false, 301));
    results.push_back(test_fused<GemmNT_Fused>("NT", 10000, 8, 32, false, true,  302));
    results.push_back(test_fused<GemmTT_Fused>("TT", 10000, 8, 32, true,  true,  303));

    printf("\n--- Test 3c: Fused transpose edge cases ---\n");
    results.push_back(test_fused<GemmNN_Fused>("NN", 200, 5, 16, false, false, 400));
    results.push_back(test_fused<GemmTN_Fused>("TN", 200, 5, 16, true,  false, 401));
    results.push_back(test_fused<GemmNT_Fused>("NT", 200, 5, 16, false, true,  402));
    results.push_back(test_fused<GemmTT_Fused>("TT", 200, 5, 16, true,  true,  403));
    results.push_back(test_fused<GemmNN_Fused>("NN", 16, 2, 4, false, false, 999));

    // ----- Test 4: Fused transpose with beta=1 (patched PermuteD load path) -----
    printf("\n--- Test 4: Fused beta=1 accumulation (M=200, N=8) ---\n");
    results.push_back(test_fused_beta1<GemmNN_Fused>("NN", 200, 8, 16, false, false, 42));
    results.push_back(test_fused_beta1<GemmTN_Fused>("TN", 200, 8, 16, true,  false, 43));
    results.push_back(test_fused_beta1<GemmNT_Fused>("NT", 200, 8, 16, false, true,  44));
    results.push_back(test_fused_beta1<GemmTT_Fused>("TT", 200, 8, 16, true,  true,  45));

    printf("\n--- Test 4b: Fused beta=1 (M=10000, N=8) ---\n");
    results.push_back(test_fused_beta1<GemmNN_Fused>("NN", 10000, 8, 32, false, false, 300));

    printf("\n--- Test 4c: Fused beta=1 edge cases ---\n");
    results.push_back(test_fused_beta1<GemmNN_Fused>("NN", 200, 5, 16, false, false, 400));
    results.push_back(test_fused_beta1<GemmNN_Fused>("NN", 16, 2, 4, false, false, 999));

    printf("\n--- Test 4d: Fused beta=1 2w (M=200, N=8) ---\n");
    results.push_back(test_fused_beta1<GemmNN_Fused_2w>("NN_2w", 200, 8, 16, false, false, 42));
    results.push_back(test_fused_beta1<GemmNN_Fused_2w>("NN_2w", 10000, 8, 32, false, false, 300));

    // ----- Timing: warp count comparison (1w/2w/4w) for Direct and Fused -----
    printf("\n=== Timing: Warp Count Comparison (NN mode, skinny-tall M>>N) ===\n\n");
    {
        const int warmup = 5, repeats = 50;
        struct BenchCase { int M; int N; int K; };
        BenchCase cases[] = {
            {10000, 8, 32},
            {50000, 8, 64},
            {100000, 8, 64},
            {10000, 3, 32},
        };

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        auto bench = [&](auto run_fn, int reps) -> double {
            for (int i = 0; i < warmup; ++i) run_fn();
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaEventRecord(start, stream));
            for (int i = 0; i < reps; ++i) run_fn();
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            return (ms / reps) * 1000.0;  // microseconds
        };

        for (auto& c : cases) {
            int M = c.M, N = c.N, K = c.K;
            int lda = K, ldb = N;

            float *d_A, *d_B, *d_C;
            CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));

            double us_d1 = bench([&]{ run_grouped_gemm<GemmNN_RowC>(M, N, K, d_A, lda, d_B, ldb, d_C, N, 1.0f, 0.0f, stream); }, repeats);
            double us_d2 = bench([&]{ run_grouped_gemm<GemmNN_RowC_2w>(M, N, K, d_A, lda, d_B, ldb, d_C, N, 1.0f, 0.0f, stream); }, repeats);
            double us_d4 = bench([&]{ run_grouped_gemm<GemmNN_RowC_4w>(M, N, K, d_A, lda, d_B, ldb, d_C, N, 1.0f, 0.0f, stream); }, repeats);
            double us_f1 = bench([&]{ run_grouped_gemm_fused<GemmNN_Fused>(M, N, K, d_A, lda, d_B, ldb, d_C, 1.0f, 0.0f, stream); }, repeats);
            double us_f2 = bench([&]{ run_grouped_gemm_fused<GemmNN_Fused_2w>(M, N, K, d_A, lda, d_B, ldb, d_C, 1.0f, 0.0f, stream); }, repeats);
            double us_f4 = bench([&]{ run_grouped_gemm_fused<GemmNN_Fused_4w>(M, N, K, d_A, lda, d_B, ldb, d_C, 1.0f, 0.0f, stream); }, repeats);

            printf("  M=%6d N=%2d K=%2d\n", M, N, K);
            printf("    Direct RowC:  1w=%6.1f us  2w=%6.1f us  4w=%6.1f us\n", us_d1, us_d2, us_d4);
            printf("    Fused:        1w=%6.1f us  2w=%6.1f us  4w=%6.1f us\n", us_f1, us_f2, us_f4);

            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaStreamDestroy(stream));
    }

    // ----- Summary -----
    printf("\n=== Results ===\n");
    int pass = 0, fail = 0;
    for (const auto& r : results) {
        const char* status = r.passed ? "PASS" : "FAIL";
        printf("  [%s] %s (max_err=%.2e)\n", status, r.name.c_str(), r.max_err);
        if (r.passed) ++pass; else ++fail;
    }
    printf("\n%d passed, %d failed out of %d tests\n", pass, fail, (int)results.size());

    return fail > 0 ? 1 : 0;
}
