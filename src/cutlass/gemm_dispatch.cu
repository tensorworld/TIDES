// CUTLASS tile shape dispatch — no CUTLASS headers needed.
// Routes prepare/fill calls to per-shape implementations via TileShape enum.

#include "general_sparse/kernels/large/cutlass_grouped_fwd.hpp"
#include "general_sparse/kernels/large/cutlass_tile_shapes.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace bsptc {

// ============================================================================
// Per-shape function declarations (defined in gemm_f32_*.cu / gemm_f64_*.cu)
// ============================================================================

// Float shapes
CutlassGemmRunnerResult prepare_gemm_f32_128x128(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_64x64(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_32x32(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_64x8(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_8x64(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_128x16(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_16x128(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_256x32(const LargeBlockWork<float>&, float, float, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f32_32x256(const LargeBlockWork<float>&, float, float, cudaStream_t);

void fill_gemm_f32_128x128(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_64x64(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_32x32(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_64x8(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_8x64(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_128x16(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_16x128(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_256x32(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);
void fill_gemm_f32_32x256(CutlassGemmRunnerResult&, const LargeBlockWork<float>&, float, float, cudaStream_t);

// Double shapes (SIMT — SM80/86)
CutlassGemmRunnerResult prepare_gemm_f64_64x64(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_32x32(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_64x8(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_8x64(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_128x16(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_16x128(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_256x32(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_32x256(const LargeBlockWork<double>&, double, double, cudaStream_t);

void fill_gemm_f64_64x64(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_32x32(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_64x8(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_8x64(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_128x16(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_16x128(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_256x32(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_32x256(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);

// Double shapes (TensorOp — SM90, DMMA f64 tensor cores)
CutlassGemmRunnerResult prepare_gemm_f64_128x128(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_64x64_tc(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_32x32_tc(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_128x16_tc(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_16x128_tc(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_256x32_tc(const LargeBlockWork<double>&, double, double, cudaStream_t);
CutlassGemmRunnerResult prepare_gemm_f64_32x256_tc(const LargeBlockWork<double>&, double, double, cudaStream_t);

void fill_gemm_f64_128x128(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_64x64_tc(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_32x32_tc(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_128x16_tc(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_16x128_tc(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_256x32_tc(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);
void fill_gemm_f64_32x256_tc(CutlassGemmRunnerResult&, const LargeBlockWork<double>&, double, double, cudaStream_t);

// ============================================================================
// SM version cache
// ============================================================================

static int get_device_sm_version() {
    static int sm_version = -1;
    if (sm_version < 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        sm_version = props.major * 10 + props.minor;
    }
    return sm_version;
}

// ============================================================================
// TileShape-aware dispatch
// ============================================================================

template <>
CutlassGemmRunnerResult prepare_cutlass_gemm_dispatch<float>(
    const LargeBlockWork<float>& work, float alpha, float beta,
    TileShape tile, cudaStream_t stream)
{
    switch (tile) {
        case TileShape::Tile_128x128: return prepare_gemm_f32_128x128(work, alpha, beta, stream);
        case TileShape::Tile_64x64: return prepare_gemm_f32_64x64(work, alpha, beta, stream);
        case TileShape::Tile_32x32: return prepare_gemm_f32_32x32(work, alpha, beta, stream);
        case TileShape::Tile_64x8:  return prepare_gemm_f32_64x8(work, alpha, beta, stream);
        case TileShape::Tile_8x64:  return prepare_gemm_f32_8x64(work, alpha, beta, stream);
        case TileShape::Tile_128x16: return prepare_gemm_f32_128x16(work, alpha, beta, stream);
        case TileShape::Tile_16x128: return prepare_gemm_f32_16x128(work, alpha, beta, stream);
        case TileShape::Tile_256x32: return prepare_gemm_f32_256x32(work, alpha, beta, stream);
        case TileShape::Tile_32x256: return prepare_gemm_f32_32x256(work, alpha, beta, stream);
    }
    throw std::runtime_error("Unsupported TileShape (float)");
}

template <>
CutlassGemmRunnerResult prepare_cutlass_gemm_dispatch<double>(
    const LargeBlockWork<double>& work, double alpha, double beta,
    TileShape tile, cudaStream_t stream)
{
    bool sm90 = get_device_sm_version() >= 90;
    switch (tile) {
        case TileShape::Tile_128x128: return sm90 ? prepare_gemm_f64_128x128(work, alpha, beta, stream)
                                                   : prepare_gemm_f64_64x64(work, alpha, beta, stream);
        case TileShape::Tile_64x64: return sm90 ? prepare_gemm_f64_64x64_tc(work, alpha, beta, stream)
                                                 : prepare_gemm_f64_64x64(work, alpha, beta, stream);
        case TileShape::Tile_32x32: return sm90 ? prepare_gemm_f64_32x32_tc(work, alpha, beta, stream)
                                                 : prepare_gemm_f64_32x32(work, alpha, beta, stream);
        case TileShape::Tile_64x8:  return prepare_gemm_f64_64x8(work, alpha, beta, stream);
        case TileShape::Tile_8x64:  return prepare_gemm_f64_8x64(work, alpha, beta, stream);
        case TileShape::Tile_128x16: return sm90 ? prepare_gemm_f64_128x16_tc(work, alpha, beta, stream)
                                                  : prepare_gemm_f64_128x16(work, alpha, beta, stream);
        case TileShape::Tile_16x128: return sm90 ? prepare_gemm_f64_16x128_tc(work, alpha, beta, stream)
                                                  : prepare_gemm_f64_16x128(work, alpha, beta, stream);
        case TileShape::Tile_256x32: return sm90 ? prepare_gemm_f64_256x32_tc(work, alpha, beta, stream)
                                                  : prepare_gemm_f64_256x32(work, alpha, beta, stream);
        case TileShape::Tile_32x256: return sm90 ? prepare_gemm_f64_32x256_tc(work, alpha, beta, stream)
                                                  : prepare_gemm_f64_32x256(work, alpha, beta, stream);
    }
    throw std::runtime_error("Unsupported TileShape (double)");
}

template <>
void fill_cutlass_args_dispatch<float>(
    CutlassGemmRunnerResult& result,
    const LargeBlockWork<float>& work, float alpha, float beta,
    TileShape tile, cudaStream_t stream)
{
    switch (tile) {
        case TileShape::Tile_128x128: fill_gemm_f32_128x128(result, work, alpha, beta, stream); return;
        case TileShape::Tile_64x64: fill_gemm_f32_64x64(result, work, alpha, beta, stream); return;
        case TileShape::Tile_32x32: fill_gemm_f32_32x32(result, work, alpha, beta, stream); return;
        case TileShape::Tile_64x8:  fill_gemm_f32_64x8(result, work, alpha, beta, stream); return;
        case TileShape::Tile_8x64:  fill_gemm_f32_8x64(result, work, alpha, beta, stream); return;
        case TileShape::Tile_128x16: fill_gemm_f32_128x16(result, work, alpha, beta, stream); return;
        case TileShape::Tile_16x128: fill_gemm_f32_16x128(result, work, alpha, beta, stream); return;
        case TileShape::Tile_256x32: fill_gemm_f32_256x32(result, work, alpha, beta, stream); return;
        case TileShape::Tile_32x256: fill_gemm_f32_32x256(result, work, alpha, beta, stream); return;
    }
    throw std::runtime_error("Unsupported TileShape (float)");
}

template <>
void fill_cutlass_args_dispatch<double>(
    CutlassGemmRunnerResult& result,
    const LargeBlockWork<double>& work, double alpha, double beta,
    TileShape tile, cudaStream_t stream)
{
    bool sm90 = get_device_sm_version() >= 90;
    switch (tile) {
        case TileShape::Tile_128x128:
            if (sm90) fill_gemm_f64_128x128(result, work, alpha, beta, stream);
            else      fill_gemm_f64_64x64(result, work, alpha, beta, stream);
            return;
        case TileShape::Tile_64x64:
            if (sm90) fill_gemm_f64_64x64_tc(result, work, alpha, beta, stream);
            else      fill_gemm_f64_64x64(result, work, alpha, beta, stream);
            return;
        case TileShape::Tile_32x32:
            if (sm90) fill_gemm_f64_32x32_tc(result, work, alpha, beta, stream);
            else      fill_gemm_f64_32x32(result, work, alpha, beta, stream);
            return;
        case TileShape::Tile_64x8:  fill_gemm_f64_64x8(result, work, alpha, beta, stream); return;
        case TileShape::Tile_8x64:  fill_gemm_f64_8x64(result, work, alpha, beta, stream); return;
        case TileShape::Tile_128x16:
            if (sm90) fill_gemm_f64_128x16_tc(result, work, alpha, beta, stream);
            else      fill_gemm_f64_128x16(result, work, alpha, beta, stream);
            return;
        case TileShape::Tile_16x128:
            if (sm90) fill_gemm_f64_16x128_tc(result, work, alpha, beta, stream);
            else      fill_gemm_f64_16x128(result, work, alpha, beta, stream);
            return;
        case TileShape::Tile_256x32:
            if (sm90) fill_gemm_f64_256x32_tc(result, work, alpha, beta, stream);
            else      fill_gemm_f64_256x32(result, work, alpha, beta, stream);
            return;
        case TileShape::Tile_32x256:
            if (sm90) fill_gemm_f64_32x256_tc(result, work, alpha, beta, stream);
            else      fill_gemm_f64_32x256(result, work, alpha, beta, stream);
            return;
    }
    throw std::runtime_error("Unsupported TileShape (double)");
}

// ============================================================================
// Utility functions
// ============================================================================

bool is_cutlass_grouped_gemm_available() {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, 0);
    if (err != cudaSuccess) {
        return false;
    }
    return props.major >= 8;
}

} // namespace bsptc
