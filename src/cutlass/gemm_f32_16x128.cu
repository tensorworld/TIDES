// Float 16x128 CUTLASS grouped GEMM (intermediate skinny-wide: N >> M)
// 4 warps (128 threads): threadblock <16,128,8>, warp <8,64,8>
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<16, 128, 8>,
    cutlass::gemm::GemmShape<8, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f32_16x128(const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f32_16x128");
}

void fill_gemm_f32_16x128(CutlassGemmRunnerResult& result, const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f32_16x128");
}

} // namespace bsptc
