// Double 64x64 CUTLASS grouped GEMM (SM90 Hopper, f64 tensor cores / DMMA)
// 4 warps (128 threads): threadblock <64,64,16>, warp <32,32,16>
#include "gemm_tu_common.inl"

namespace bsptc {
namespace {

using Cfg = gemm_tu::GroupedGemmConfig<double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 4>,
    3>;

} // anonymous namespace

CutlassGemmRunnerResult prepare_gemm_f64_64x64_tc(const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    return gemm_tu::prepare_grouped<Cfg>(work, alpha, beta, stream, "f64_64x64_tc");
}

void fill_gemm_f64_64x64_tc(CutlassGemmRunnerResult& result, const LargeBlockWork<double>& work, double alpha, double beta, cudaStream_t stream) {
    gemm_tu::fill_grouped<Cfg>(result, work, alpha, beta, stream, "f64_64x64_tc");
}

} // namespace bsptc
