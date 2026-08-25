// Shared prelude for CUTLASS grouped-GEMM instantiation TUs (src/cutlass/gemm_*.cu).
// Each TU instantiates one (dtype, tile shape) configuration so they compile in
// parallel; everything they share lives here.
// CRITICAL: CUTLASS headers MUST be included FIRST (before any CUDA/cuTENSOR).
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "general_sparse/kernels/large/cutlass_gemm_impl.hpp"

#include <string>

namespace bsptc {
namespace gemm_tu {

// TransposePermute: fused epilogue transpose via CUTLASS PermuteDLayout.
// See tests/test_cutlass_skinny.cu for detailed derivation.
//
// With ColumnMajor-C + kInternalTranspose:
//   M_int = N_ext, N_int = M_ext
//   Epilogue col = M_ext coord, row = N_ext coord
//   Constructor: contiguous = N_int = M_ext, strided = M_int = N_ext
//   offset = col * N_ext + row = m * N + n (row-major output)
//
// Beta=1 accumulation requires the patched CUTLASS predicated_tile_iterator.h
// where the load path also uses permute_layout_() when PermuteD is true.
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

// Fused-transpose TUs override ldc from N to M for correct internal
// predication (the kernel sees the transposed [N x M] problem). The CUTLASS
// arg-staging loop generates ldc AND ldd from the desc's ldc field, so
// overriding the desc is sufficient.
template <typename Element>
LargeBlockWork<Element> override_ldc_to_M(const LargeBlockWork<Element>& work) {
    LargeBlockWork<Element> fused_work = work;
    for (auto& prob : fused_work.problems)
        prob.ldc = prob.M;
    return fused_work;
}

// One grouped-GEMM configuration: the four A/B layout combinations (NN/TN/
// NT/TT) of a DefaultGemmGrouped kernel with the given element type, opclass,
// arch, tile shapes, and stage count.  LayoutC = ColumnMajor together with
// PermuteD = TransposePermute selects the fused-transpose epilogue
// (kInternalTranspose swaps M/N internally).
template <typename Element, typename OpClass, typename Arch,
          typename TBShape, typename WarpShape, typename InstShape, int Stages,
          typename LayoutC = cutlass::layout::RowMajor,
          typename PermuteD = cutlass::layout::NoPermute>
struct GroupedGemmConfig {
    template <typename LayoutA, typename LayoutB>
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        Element, LayoutA, cutlass::ComplexTransform::kNone, 1,
        Element, LayoutB, cutlass::ComplexTransform::kNone, 1,
        Element, LayoutC, Element,
        OpClass, Arch,
        TBShape, WarpShape, InstShape,
        cutlass::epilogue::thread::LinearCombination<Element, 1, Element, Element>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass, Arch, Element, Element, Element, Element>::Operator,
        cutlass::gemm::SharedMemoryClearOption::kNone,
        PermuteD
    >::GemmKernel;

    using RM = cutlass::layout::RowMajor;
    using CM = cutlass::layout::ColumnMajor;
    using GemmNN = cutlass::gemm::device::GemmGrouped<Kernel<RM, RM>>;
    using GemmTN = cutlass::gemm::device::GemmGrouped<Kernel<CM, RM>>;
    using GemmNT = cutlass::gemm::device::GemmGrouped<Kernel<RM, CM>>;
    using GemmTT = cutlass::gemm::device::GemmGrouped<Kernel<CM, CM>>;
};

template <typename Cfg, typename Element>
CutlassGemmRunnerResult prepare_grouped(const LargeBlockWork<Element>& work,
                                        Element alpha, Element beta,
                                        cudaStream_t stream, const char* name) {
    switch (work.gemm_mode) {
        case GemmMode::NN: return prepare_cutlass_gemm_impl<typename Cfg::GemmNN>(work, alpha, beta, stream);
        case GemmMode::TN: return prepare_cutlass_gemm_impl<typename Cfg::GemmTN>(work, alpha, beta, stream);
        case GemmMode::NT: return prepare_cutlass_gemm_impl<typename Cfg::GemmNT>(work, alpha, beta, stream);
        case GemmMode::TT: return prepare_cutlass_gemm_impl<typename Cfg::GemmTT>(work, alpha, beta, stream);
    }
    throw std::runtime_error(std::string("Unsupported GEMM mode (") + name + ")");
}

template <typename Cfg, typename Element>
void fill_grouped(CutlassGemmRunnerResult& result, const LargeBlockWork<Element>& work,
                  Element alpha, Element beta, cudaStream_t stream, const char* name) {
    switch (work.gemm_mode) {
        case GemmMode::NN: fill_cutlass_args_impl<typename Cfg::GemmNN>(result, work, alpha, beta, stream); return;
        case GemmMode::TN: fill_cutlass_args_impl<typename Cfg::GemmTN>(result, work, alpha, beta, stream); return;
        case GemmMode::NT: fill_cutlass_args_impl<typename Cfg::GemmNT>(result, work, alpha, beta, stream); return;
        case GemmMode::TT: fill_cutlass_args_impl<typename Cfg::GemmTT>(result, work, alpha, beta, stream); return;
    }
    throw std::runtime_error(std::string("Unsupported GEMM mode (") + name + ")");
}

} // namespace gemm_tu
} // namespace bsptc
