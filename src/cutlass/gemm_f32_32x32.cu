// Float 32x32 CUTLASS grouped GEMM (small square, M,N in [9, 31])
// CRITICAL: CUTLASS headers MUST be included FIRST
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "general_sparse/kernels/large/cutlass_gemm_impl.hpp"

namespace bsptc {
namespace {

using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

template <typename LayoutA, typename LayoutB>
using GemmKernel_f32_32x32 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1,
    float, LayoutB, cutlass::ComplexTransform::kNone, 1,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<16, 16, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmNN = cutlass::gemm::device::GemmGrouped<GemmKernel_f32_32x32<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmTN = cutlass::gemm::device::GemmGrouped<GemmKernel_f32_32x32<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>>;
using GemmNT = cutlass::gemm::device::GemmGrouped<GemmKernel_f32_32x32<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>>;
using GemmTT = cutlass::gemm::device::GemmGrouped<GemmKernel_f32_32x32<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f32_32x32(const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    switch (work.gemm_mode) {
        case GemmMode::NN: return prepare_cutlass_gemm_impl<GemmNN>(work, alpha, beta, stream);
        case GemmMode::TN: return prepare_cutlass_gemm_impl<GemmTN>(work, alpha, beta, stream);
        case GemmMode::NT: return prepare_cutlass_gemm_impl<GemmNT>(work, alpha, beta, stream);
        case GemmMode::TT: return prepare_cutlass_gemm_impl<GemmTT>(work, alpha, beta, stream);
    }
    throw std::runtime_error("Unsupported GEMM mode (f32_32x32)");
}

void fill_gemm_f32_32x32(CutlassGemmRunnerResult& result, const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    switch (work.gemm_mode) {
        case GemmMode::NN: fill_cutlass_args_impl<GemmNN>(result, work, alpha, beta, stream); return;
        case GemmMode::TN: fill_cutlass_args_impl<GemmTN>(result, work, alpha, beta, stream); return;
        case GemmMode::NT: fill_cutlass_args_impl<GemmNT>(result, work, alpha, beta, stream); return;
        case GemmMode::TT: fill_cutlass_args_impl<GemmTT>(result, work, alpha, beta, stream); return;
    }
    throw std::runtime_error("Unsupported GEMM mode (f32_32x32)");
}

} // namespace bsptc
