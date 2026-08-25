// Float 256x32 CUTLASS grouped GEMM (intermediate tall: M >> N, min(M,N) >= 32)
// min(M,N) = 32 is large enough for direct RowMajor-C — no fused transpose needed.
// 8 warps (256 threads): threadblock <256,32,8>, warp <64,16,8>
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 32, 8>,
    cutlass::gemm::GemmShape<64, 16, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f32_256x32(const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f32_256x32");
}

void fill_gemm_f32_256x32(CutlassGemmRunnerResult& result, const LargeBlockWork<float>& work, float alpha, float beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f32_256x32");
}

} // namespace bsptc
