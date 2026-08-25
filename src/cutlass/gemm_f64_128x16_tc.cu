// Double 128x16 fused-transpose CUTLASS grouped GEMM (SM90 Hopper, f64 tensor cores / DMMA)
// ColumnMajor-C triggers kInternalTranspose: swaps M/N internally so the
// <16,128,16> tile sees [N x M] (skinny-wide internally). TransposePermute writes
// row-major [M x N] output directly in the epilogue — no separate transpose.
// 4 warps (128 threads): threadblock <16,128,16>, warp <16,32,16>
//
// ldc is overridden to M (not N) for correct internal predication.
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<16, 128, 16>,
    cutlass::gemm::GemmShape<16, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    3,
    cutlass::layout::ColumnMajor, gemm_tu::TransposePermute>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_128x16_tc(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(gemm_tu::override_ldc_to_M(work), alpha, beta, stream, "f64_fused_128x16_tc");
}

void fill_gemm_f64_128x16_tc(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, gemm_tu::override_ldc_to_M(work), alpha, beta, stream, "f64_fused_128x16_tc");
}

} // namespace bsptc
