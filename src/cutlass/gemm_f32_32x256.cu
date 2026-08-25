// Float 32x256 CUTLASS grouped GEMM (intermediate skinny-wide: N >> M)
// 8 warps (256 threads): threadblock <32,256,8>, warp <16,64,8>
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 256, 8>,
    cutlass::gemm::GemmShape<16, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f32_32x256(const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f32_32x256");
}

void fill_gemm_f32_32x256(CutlassGemmRunnerResult& result, const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f32_32x256");
}

} // namespace bsptc
