// Double 128x16 fused-transpose CUTLASS grouped GEMM (SM80 SIMT, skinny-tall: M >> N)
// ColumnMajor-C triggers kInternalTranspose: swaps M/N internally so the
// <16,128,8> tile sees [N x M] (skinny-wide internally). TransposePermute writes
// row-major [M x N] output directly in the epilogue — no separate transpose.
// 4 warps (128 threads): threadblock <16,128,8>, warp <8,64,8>
//
// ldc is overridden to M (not N) for correct internal predication.
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "general_sparse/kernels/large/cutlass_gemm_impl.hpp"

namespace bsptc {
namespace {

using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

class TransposePermute : public cutlass::layout::PermuteBase {
private:
    Index N_orig_;
public:
    CUTLASS_HOST_DEVICE
    TransposePermute(cutlass::layout::PitchLinearCoord extent, Index /*stride*/) {
        N_orig_ = extent.strided();
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(cutlass::layout::PitchLinearCoord coord) const {
        return LongIndex(coord.contiguous()) * LongIndex(N_orig_) + LongIndex(coord.strided());
    }
};

template <typename LayoutA, typename LayoutB>
using GemmKernel_f64_fused_128x16 = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, LayoutA, cutlass::ComplexTransform::kNone, 1,
    double, LayoutB, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::ColumnMajor, double,  // triggers kInternalTranspose
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<16, 128, 8>,
    cutlass::gemm::GemmShape<8, 64, 8>,  // 4 warps
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2, GroupScheduleMode::kDeviceOnly,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
        double, double, double, double>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    TransposePermute
>::GemmKernel;

using GemmNN = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_fused_128x16<cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;
using GemmTN = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_fused_128x16<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>>;
using GemmNT = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_fused_128x16<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>>;
using GemmTT = cutlass::gemm::device::GemmGrouped<GemmKernel_f64_fused_128x16<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>>;

LargeBlockWork<double> override_ldc_to_M(const LargeBlockWork<double>& work) {
    LargeBlockWork<double> fused_work = work;
    size_t idx = 0;
    for (const auto& prob : fused_work.problems) {
        for (int64_t b = 0; b < prob.batch_size; ++b) {
            fused_work.ldc[idx] = static_cast<int64_t>(prob.M);
            fused_work.ldd[idx] = static_cast<int64_t>(prob.M);
            ++idx;
        }
    }
    return fused_work;
}

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_128x16(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    auto fused_work = override_ldc_to_M(work);
    switch (work.gemm_mode) {
        case GemmMode::NN: return prepare_cutlass_gemm_impl<GemmNN>(fused_work, alpha, beta, stream);
        case GemmMode::TN: return prepare_cutlass_gemm_impl<GemmTN>(fused_work, alpha, beta, stream);
        case GemmMode::NT: return prepare_cutlass_gemm_impl<GemmNT>(fused_work, alpha, beta, stream);
        case GemmMode::TT: return prepare_cutlass_gemm_impl<GemmTT>(fused_work, alpha, beta, stream);
    }
    throw std::runtime_error("Unsupported GEMM mode (f64_fused_128x16)");
}

void fill_gemm_f64_128x16(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    auto fused_work = override_ldc_to_M(work);
    switch (work.gemm_mode) {
        case GemmMode::NN: fill_cutlass_args_impl<GemmNN>(result, fused_work, alpha, beta, stream); return;
        case GemmMode::TN: fill_cutlass_args_impl<GemmTN>(result, fused_work, alpha, beta, stream); return;
        case GemmMode::NT: fill_cutlass_args_impl<GemmNT>(result, fused_work, alpha, beta, stream); return;
        case GemmMode::TT: fill_cutlass_args_impl<GemmTT>(result, fused_work, alpha, beta, stream); return;
    }
    throw std::runtime_error("Unsupported GEMM mode (f64_fused_128x16)");
}

} // namespace bsptc
