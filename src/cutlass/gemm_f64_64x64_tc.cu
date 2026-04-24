// Double 64x64 CUTLASS grouped GEMM (SM90 Hopper, f64 tensor cores / DMMA)
// 4 warps (128 threads): threadblock <64,64,16>, warp <32,32,16>
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
using GemmKernel_f64_64x64_tc = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, LayoutA, cutlass::ComplexTransform::kNone, 1,
    double, LayoutB, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3, GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmNN = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_64x64_tc<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmTN = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_64x64_tc<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>>;
using GemmNT = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_64x64_tc<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>>;
using GemmTT = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_64x64_tc<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_64x64_tc(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    switch (work.gemm_mode) {
        case GemmMode::NN: return prepare_cutlass_gemm_impl<GemmNN>(work, alpha, beta, stream);
        case GemmMode::TN: return prepare_cutlass_gemm_impl<GemmTN>(work, alpha, beta, stream);
        case GemmMode::NT: return prepare_cutlass_gemm_impl<GemmNT>(work, alpha, beta, stream);
        case GemmMode::TT: return prepare_cutlass_gemm_impl<GemmTT>(work, alpha, beta, stream);
    }
    throw std::runtime_error("Unsupported GEMM mode (f64_64x64_tc)");
}

void fill_gemm_f64_64x64_tc(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    switch (work.gemm_mode) {
        case GemmMode::NN: fill_cutlass_args_impl<GemmNN>(result, work, alpha, beta, stream); return;
        case GemmMode::TN: fill_cutlass_args_impl<GemmTN>(result, work, alpha, beta, stream); return;
        case GemmMode::NT: fill_cutlass_args_impl<GemmNT>(result, work, alpha, beta, stream); return;
        case GemmMode::TT: fill_cutlass_args_impl<GemmTT>(result, work, alpha, beta, stream); return;
    }
    throw std::runtime_error("Unsupported GEMM mode (f64_64x64_tc)");
}

} // namespace bsptc
