// Float 64x64 CUTLASS grouped GEMM (square, min(M,N) >= 16)
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f32_64x64(const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f32_64x64");
}

void fill_gemm_f32_64x64(CutlassGemmRunnerResult& result, const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f32_64x64");
}

} // namespace bsptc
