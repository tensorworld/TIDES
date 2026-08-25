// Double 128x16 fused-transpose CUTLASS grouped GEMM (SM80 SIMT, skinny-tall: M >> N)
// ColumnMajor-C triggers kInternalTranspose: swaps M/N internally so the
// <16,128,8> tile sees [N x M] (skinny-wide internally). TransposePermute writes
// row-major [M x N] output directly in the epilogue — no separate transpose.
// 4 warps (128 threads): threadblock <16,128,8>, warp <8,64,8>
//
// ldc is overridden to M (not N) for correct internal predication.
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<double,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<16, 128, 8>,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2,
    cutlass::layout::ColumnMajor, gemm_tu::TransposePermute>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_128x16(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(gemm_tu::override_ldc_to_M(work), alpha, beta, stream, "f64_fused_128x16");
}

void fill_gemm_f64_128x16(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, gemm_tu::override_ldc_to_M(work), alpha, beta, stream, "f64_fused_128x16");
}

} // namespace bsptc
