// cuBLAS grouped GEMM backend for the Large tier.
//
// Mirrors src/cutlass/gemm_*.cu: prepare/fill stage arguments into the pooled
// device+pinned buffers and hand back a run(stream) closure.  Contains no
// CUTLASS headers (cuBLAS and CUTLASS are kept in separate TUs for the same
// reason cuTENSOR is).
//
// ---------------------------------------------------------------------------
// Row-major -> cuBLAS column-major mapping
// ---------------------------------------------------------------------------
// The pipeline's GEMM problems are row-major: C[MxN] (ldc=N) = op(A[MxK]) *
// op(B[KxN]), where GemmMode picks each operand's storage order and the leading
// dims follow BLAS convention (lda = A_transposed ? M : K, ldb = B_transposed ?
// K : N).
//
// cuBLAS is column-major.  Reading the row-major C buffer as column-major with
// leading dimension ldc yields C^T (an NxM matrix), and
//
//     C^T = (A B)^T = B^T A^T
//
// so a single cuBLAS call computes the desired result with the operands swapped
// and M/N exchanged.  Reading a row-major operand as column-major already
// transposes it, which folds into the op flags:
//
//     first  operand := B, op = N if B is row-major else T,  ld = ldb
//     second operand := A, op = N if A is row-major else T,  ld = lda
//     m = N,  n = M,  k = K,  C unchanged with ld = ldc
//
// No data movement, and C comes out row-major exactly as the accumulation and
// reduce stages expect (this is why the fused-transpose epilogue that the skinny
// CUTLASS tiles need has no counterpart here).
//
// One cuBLAS "group" per GemmProblemDesc: M/N/K and all leading dims are uniform
// within a desc, and its batch_size is the group size.  The A/B/C pointer arrays
// are already batch-expanded in desc order, which is exactly the layout
// cublas*gemmGroupedBatched expects.

#include "general_sparse/kernels/large/cublas_grouped_fwd.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bsptc {

namespace {

static_assert(sizeof(cublasOperation_t) == sizeof(int),
              "cublas arg-buffer sizing assumes cublasOperation_t is int-sized");

// Region layout inside the pooled buffers.
// Device: the three pointer arrays the kernel dereferences.
// Host:   a staging mirror of those three (single H2D), followed by the
//         per-group scalar arrays, which cuBLAS reads from host memory at call
//         time and which therefore never leave the pinned buffer.
constexpr int kDevRegions = 3;   // Aarray, Barray, Carray
constexpr int kHostScalars = 11; // transa, transb, m, n, k, lda, ldb, ldc,
                                 // group_size, alpha, beta

void dev_region_sizes(int count, size_t (&s)[kDevRegions]) {
    const size_t n = static_cast<size_t>(count);
    s[0] = s[1] = s[2] = n * sizeof(void*);
}

// Scalars are per-group; group_count <= count, so `count` is a safe bound.
// alpha/beta are sized for the widest element type so the sizing is
// type-independent (allocate_cutlass_args has no element type).
void host_scalar_sizes(int count, size_t (&s)[kHostScalars]) {
    const size_t n = static_cast<size_t>(count);
    for (int i = 0; i < 9; ++i) s[i] = n * sizeof(int);
    s[9] = n * sizeof(double);   // alpha
    s[10] = n * sizeof(double);  // beta
}

cublasHandle_t get_cublas_handle() {
    // One process-wide handle.  A cuBLAS handle is not a stream and holds no
    // per-stream state (the stream is set per call), so caching it does not
    // violate the "streams are process resources" invariant.
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS: cublasCreate failed");
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        // Default to strict IEEE FP32: TF32 would change the numerics relative
        // to the CUTLASS SIMT tiles and to the cuTENSOR baseline, so it is an
        // explicit opt-in (BSPTC_CUBLAS_TF32=1) rather than a silent default.
        const char* tf32 = std::getenv("BSPTC_CUBLAS_TF32");
        cublasSetMathMode(handle, (tf32 && tf32[0] == '1')
                                      ? CUBLAS_TF32_TENSOR_OP_MATH
                                      : CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

const char* cublas_status_name(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "LICENSE_ERROR";
        default:                             return "UNKNOWN";
    }
}

// Per-call argument view resolved into the pooled buffers.
struct CublasArgs {
    cublasOperation_t* transa;
    cublasOperation_t* transb;
    int* m;
    int* n;
    int* k;
    int* lda;
    int* ldb;
    int* ldc;
    int* group_size;
    void* alpha;
    void* beta;
    void** d_Aarray;
    void** d_Barray;
    void** d_Carray;
    int group_count;
};

// Resolve the device + host regions and fill everything that does not depend on
// the operand base pointers.  Returns the total device bytes to copy H2D.
template <typename T>
size_t stage_cublas_args(CutlassGemmRunnerResult& result,
                         const LargeBlockWork<T>& work,
                         T alpha, T beta,
                         CublasArgs& out)
{
    const int count = static_cast<int>(work.num_gemm_problems);

    size_t d_sizes[kDevRegions];
    dev_region_sizes(count, d_sizes);
    const size_t d_total = total_allocation_size(d_sizes);

    void* d_ptrs[kDevRegions];
    resolve_pointers(result.d_args_buf, d_sizes, d_ptrs);
    void* h_ptrs[kDevRegions];
    resolve_pointers(result.h_args_buf, d_sizes, h_ptrs);

    // Host-only scalars live past the H2D staging mirror.
    size_t s_sizes[kHostScalars];
    host_scalar_sizes(count, s_sizes);
    void* s_ptrs[kHostScalars];
    resolve_pointers(static_cast<char*>(result.h_args_buf) + d_total,
                     s_sizes, s_ptrs);

    const GemmMode mode = work.gemm_mode;
    // Reading a row-major operand as column-major transposes it; see the header
    // comment.  Operand order is swapped, so `transa` describes B and `transb` A.
    const cublasOperation_t op_for_B =
        gemm_mode_b_transposed(mode) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_for_A =
        gemm_mode_a_transposed(mode) ? CUBLAS_OP_T : CUBLAS_OP_N;

    out.transa = static_cast<cublasOperation_t*>(s_ptrs[0]);
    out.transb = static_cast<cublasOperation_t*>(s_ptrs[1]);
    out.m = static_cast<int*>(s_ptrs[2]);
    out.n = static_cast<int*>(s_ptrs[3]);
    out.k = static_cast<int*>(s_ptrs[4]);
    out.lda = static_cast<int*>(s_ptrs[5]);
    out.ldb = static_cast<int*>(s_ptrs[6]);
    out.ldc = static_cast<int*>(s_ptrs[7]);
    out.group_size = static_cast<int*>(s_ptrs[8]);
    out.alpha = s_ptrs[9];
    out.beta = s_ptrs[10];

    auto* alpha_arr = static_cast<T*>(out.alpha);
    auto* beta_arr = static_cast<T*>(out.beta);

    const int group_count = static_cast<int>(work.problems.size());
    for (int g = 0; g < group_count; ++g) {
        const auto& p = work.problems[static_cast<size_t>(g)];
        out.transa[g] = op_for_B;
        out.transb[g] = op_for_A;
        out.m[g] = static_cast<int>(p.N);   // swapped: cuBLAS computes C^T
        out.n[g] = static_cast<int>(p.M);
        out.k[g] = static_cast<int>(p.K);
        out.lda[g] = static_cast<int>(p.ldb);  // first operand is B
        out.ldb[g] = static_cast<int>(p.lda);  // second operand is A
        out.ldc[g] = static_cast<int>(p.ldc);
        out.group_size[g] = static_cast<int>(p.batch_size);
        alpha_arr[g] = alpha;
        beta_arr[g] = beta;
    }
    out.group_count = group_count;

    // Aarray receives B and Barray receives A at call time, but the staging
    // regions stay keyed to the *operand* so the post-transpose pointer patch
    // (patch_ab_pointers) keeps writing A into the A region.
    result.d_ptr_A_loc = d_ptrs[0];
    result.d_ptr_B_loc = d_ptrs[1];
    result.h_ptr_A_loc = h_ptrs[0];
    result.h_ptr_B_loc = h_ptrs[1];
    result.ptr_array_bytes = d_sizes[0];

    std::memcpy(h_ptrs[0], work.ptr_A.data(), d_sizes[0]);
    std::memcpy(h_ptrs[1], work.ptr_B.data(), d_sizes[1]);
    std::memcpy(h_ptrs[2], work.ptr_C.data(), d_sizes[2]);

    out.d_Aarray = static_cast<void**>(d_ptrs[0]);  // A pointers
    out.d_Barray = static_cast<void**>(d_ptrs[1]);  // B pointers
    out.d_Carray = static_cast<void**>(d_ptrs[2]);

    return d_total;
}

template <typename T>
void launch_grouped(const CublasArgs& a, cudaStream_t s) {
    cublasHandle_t h = get_cublas_handle();
    if (cublasSetStream(h, s) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS: cublasSetStream failed");

    cublasStatus_t st;
    // Operand swap: cuBLAS Aarray := our B pointers, Barray := our A pointers.
    if constexpr (std::is_same_v<T, float>) {
        st = cublasSgemmGroupedBatched(
            h, a.transa, a.transb, a.m, a.n, a.k,
            static_cast<const float*>(a.alpha),
            const_cast<const float* const*>(reinterpret_cast<float**>(a.d_Barray)), a.lda,
            const_cast<const float* const*>(reinterpret_cast<float**>(a.d_Aarray)), a.ldb,
            static_cast<const float*>(a.beta),
            reinterpret_cast<float* const*>(a.d_Carray), a.ldc,
            a.group_count, a.group_size);
    } else {
        st = cublasDgemmGroupedBatched(
            h, a.transa, a.transb, a.m, a.n, a.k,
            static_cast<const double*>(a.alpha),
            const_cast<const double* const*>(reinterpret_cast<double**>(a.d_Barray)), a.lda,
            const_cast<const double* const*>(reinterpret_cast<double**>(a.d_Aarray)), a.ldb,
            static_cast<const double*>(a.beta),
            reinterpret_cast<double* const*>(a.d_Carray), a.ldc,
            a.group_count, a.group_size);
    }
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS grouped GEMM failed: ") +
                                 cublas_status_name(st));
}

} // namespace

size_t cublas_args_device_bytes(int count) {
    if (count <= 0) return 0;
    size_t s[kDevRegions];
    dev_region_sizes(count, s);
    return total_allocation_size(s);
}

size_t cublas_args_host_bytes(int count) {
    if (count <= 0) return 0;
    size_t s[kHostScalars];
    host_scalar_sizes(count, s);
    // staging mirror of the device regions + the host-only scalar block, with a
    // full alignment pad between the two so the scalars start on a boundary.
    return cublas_args_device_bytes(count) + GEMM_ARG_ALIGN +
           total_allocation_size(s);
}

bool is_cublas_grouped_gemm_available() {
    static int cached = -1;
    if (cached < 0) {
        cublasHandle_t h = nullptr;
        cached = (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) ? 1 : 0;
        if (h) cublasDestroy(h);
    }
    return cached == 1;
}

template <typename T>
CutlassGemmRunnerResult prepare_cublas_gemm(
    const LargeBlockWork<T>& work, T alpha, T beta, cudaStream_t stream)
{
    CutlassGemmRunnerResult result;
    const int count = static_cast<int>(work.num_gemm_problems);
    if (count <= 0) return result;

    result.d_args_buf = device_args_pool().acquire(cublas_args_device_bytes(count));
    result.h_args_buf = pinned_args_pool().acquire(cublas_args_host_bytes(count));

    CublasArgs args{};
    const size_t d_total = stage_cublas_args(result, work, alpha, beta, args);

    check_cuda(cudaMemcpyAsync(result.d_args_buf, result.h_args_buf, d_total,
                               cudaMemcpyHostToDevice, stream),
               "prepare_cublas_gemm: memcpy ptr arrays");

    result.run = [args](cudaStream_t s) { launch_grouped<T>(args, s); };
    return result;
}

template <typename T>
void fill_cublas_args(CutlassGemmRunnerResult& result,
                      const LargeBlockWork<T>& work,
                      T alpha, T beta, cudaStream_t stream)
{
    const int count = static_cast<int>(work.num_gemm_problems);
    if (count <= 0) { result.run = nullptr; return; }

    CublasArgs args{};
    const size_t d_total = stage_cublas_args(result, work, alpha, beta, args);

    check_cuda(cudaMemcpyAsync(result.d_args_buf, result.h_args_buf, d_total,
                               cudaMemcpyHostToDevice, stream),
               "fill_cublas_args: memcpy ptr arrays");

    result.run = [args](cudaStream_t s) { launch_grouped<T>(args, s); };
}

template CutlassGemmRunnerResult prepare_cublas_gemm<float>(
    const LargeBlockWork<float>&, float, float, cudaStream_t);
template CutlassGemmRunnerResult prepare_cublas_gemm<double>(
    const LargeBlockWork<double>&, double, double, cudaStream_t);
template void fill_cublas_args<float>(
    CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
template void fill_cublas_args<double>(
    CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);

} // namespace bsptc
