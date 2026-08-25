// Double 128x128 CUTLASS grouped GEMM (SM90 Hopper, f64 tensor cores / DMMA)
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    3>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_128x128(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f64_128x128");
}

void fill_gemm_f64_128x128(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f64_128x128");
}

} // namespace bsptc
