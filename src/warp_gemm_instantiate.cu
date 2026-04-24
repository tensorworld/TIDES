#include "general_sparse/kernels/small/warp_gemm.cuh"
#include "general_sparse/device_props.hpp"

namespace bsptc {

// ============================================================================
// Explicit instantiations for all 16 variants × 2 types (float, double)
// variant = (INT_LARGE << 3) | (INT_K << 2) | (CONTIG_K_A << 1) | CONTIG_K_B
// ============================================================================

#define INSTANTIATE_WARP(T, IL, IK, CKA, CKB) \
    template __global__ void warp_gemm_kernel<T, IL, IK, CKA, CKB>( \
        const WarpShape< \
            std::conditional_t<IL, IntDim, StdDim>, \
            std::conditional_t<IK, IntK, StdK> \
        >* __restrict__, \
        const WarpMem* __restrict__, \
        const int32_t* __restrict__, \
        int, int);

// float instantiations (16 variants)
INSTANTIATE_WARP(float, false, false, false, false)
INSTANTIATE_WARP(float, false, false, false, true)
INSTANTIATE_WARP(float, false, false, true,  false)
INSTANTIATE_WARP(float, false, false, true,  true)
INSTANTIATE_WARP(float, false, true,  false, false)
INSTANTIATE_WARP(float, false, true,  false, true)
INSTANTIATE_WARP(float, false, true,  true,  false)
INSTANTIATE_WARP(float, false, true,  true,  true)
INSTANTIATE_WARP(float, true,  false, false, false)
INSTANTIATE_WARP(float, true,  false, false, true)
INSTANTIATE_WARP(float, true,  false, true,  false)
INSTANTIATE_WARP(float, true,  false, true,  true)
INSTANTIATE_WARP(float, true,  true,  false, false)
INSTANTIATE_WARP(float, true,  true,  false, true)
INSTANTIATE_WARP(float, true,  true,  true,  false)
INSTANTIATE_WARP(float, true,  true,  true,  true)

// double instantiations (16 variants)
INSTANTIATE_WARP(double, false, false, false, false)
INSTANTIATE_WARP(double, false, false, false, true)
INSTANTIATE_WARP(double, false, false, true,  false)
INSTANTIATE_WARP(double, false, false, true,  true)
INSTANTIATE_WARP(double, false, true,  false, false)
INSTANTIATE_WARP(double, false, true,  false, true)
INSTANTIATE_WARP(double, false, true,  true,  false)
INSTANTIATE_WARP(double, false, true,  true,  true)
INSTANTIATE_WARP(double, true,  false, false, false)
INSTANTIATE_WARP(double, true,  false, false, true)
INSTANTIATE_WARP(double, true,  false, true,  false)
INSTANTIATE_WARP(double, true,  false, true,  true)
INSTANTIATE_WARP(double, true,  true,  false, false)
INSTANTIATE_WARP(double, true,  true,  false, true)
INSTANTIATE_WARP(double, true,  true,  true,  false)
INSTANTIATE_WARP(double, true,  true,  true,  true)

#undef INSTANTIATE_WARP

// ============================================================================
// Launch function implementation
// ============================================================================

template <typename T>
void launch_warp_gemm(const void* d_shapes, const WarpMem* d_mems,
                      const int32_t* d_warp_prefix, int num_descs,
                      int total_warps, int variant, cudaStream_t stream)
{
    if (total_warps == 0) return;
    constexpr int BLOCK_SIZE = 128;  // 4 warps per block
    int needed = (total_warps * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Cap grid for stride-loop execution: enough blocks to fill GPU,
    // each warp processes multiple work items via the kernel's outer loop.
    int grid = std::min(needed, get_num_sms() * 12);

    #define LAUNCH_CASE(V, IL, IK, CKA, CKB) \
        case V: { \
            using ST = WarpShape<std::conditional_t<IL, IntDim, StdDim>, \
                                 std::conditional_t<IK, IntK, StdK>>; \
            warp_gemm_kernel<T, IL, IK, CKA, CKB> \
                <<<grid, BLOCK_SIZE, 0, stream>>>( \
                    static_cast<const ST*>(d_shapes), d_mems, \
                    d_warp_prefix, num_descs, total_warps); \
            break; \
        }

    switch (variant) {
        // variant = (INT_LARGE << 3) | (INT_K << 2) | (CONTIG_K_A << 1) | CONTIG_K_B
        LAUNCH_CASE( 0, false, false, false, false)
        LAUNCH_CASE( 1, false, false, false, true)
        LAUNCH_CASE( 2, false, false, true,  false)
        LAUNCH_CASE( 3, false, false, true,  true)
        LAUNCH_CASE( 4, false, true,  false, false)
        LAUNCH_CASE( 5, false, true,  false, true)
        LAUNCH_CASE( 6, false, true,  true,  false)
        LAUNCH_CASE( 7, false, true,  true,  true)
        LAUNCH_CASE( 8, true,  false, false, false)
        LAUNCH_CASE( 9, true,  false, false, true)
        LAUNCH_CASE(10, true,  false, true,  false)
        LAUNCH_CASE(11, true,  false, true,  true)
        LAUNCH_CASE(12, true,  true,  false, false)
        LAUNCH_CASE(13, true,  true,  false, true)
        LAUNCH_CASE(14, true,  true,  true,  false)
        LAUNCH_CASE(15, true,  true,  true,  true)
    }
    #undef LAUNCH_CASE
}

// Explicit instantiations of launch function
template void launch_warp_gemm<float>(
    const void* d_shapes, const WarpMem* d_mems,
    const int32_t* d_warp_prefix, int num_descs,
    int total_warps, int variant, cudaStream_t stream);

template void launch_warp_gemm<double>(
    const void* d_shapes, const WarpMem* d_mems,
    const int32_t* d_warp_prefix, int num_descs,
    int total_warps, int variant, cudaStream_t stream);

} // namespace bsptc
