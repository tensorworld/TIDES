// Double 64x64 CUTLASS grouped GEMM (square, min(M,N) >= 16)
// 8 warps (256 threads): threadblock <64,64,8>, warp <32,16,8>
// Warp 32x16 = 512 f64 accumulators = 128 regs — feasible on SM80 (max 255)
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<double,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 16, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    2>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_64x64(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f64_64x64");
}

void fill_gemm_f64_64x64(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f64_64x64");
}

} // namespace bsptc
