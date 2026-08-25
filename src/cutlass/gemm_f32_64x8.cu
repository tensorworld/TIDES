// Float 64x8 fused-transpose CUTLASS grouped GEMM (skinny-tall: M >> N)
// ColumnMajor-C triggers kInternalTranspose: swaps M/N internally so the
// <8,64,8> tile sees [N x M] (skinny-wide internally). TransposePermute writes
// row-major [M x N] output directly in the epilogue — no separate transpose.
// 2 warps (64 threads): threadblock <8,64,8>, warp <8,32,8>
//
// ldc is overridden to M (not N) for correct internal predication.
//
// Beta=1 accumulation requires the patched CUTLASS predicated_tile_iterator.h
// where the load path also uses permute_layout_() when PermuteD is true.
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<8, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2,
    cutlass::layout::ColumnMajor, gemm_tu::TransposePermute>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f32_64x8(const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(gemm_tu::override_ldc_to_M(work), alpha, beta, stream, "f32_fused_64x8");
}

void fill_gemm_f32_64x8(CutlassGemmRunnerResult& result, const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, gemm_tu::override_ldc_to_M(work), alpha, beta, stream, "f32_fused_64x8");
}

} // namespace bsptc
