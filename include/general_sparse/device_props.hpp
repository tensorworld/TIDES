#pragma once

#include <cuda_runtime.h>
#include <algorithm>

namespace bsptc {

// Cached SM count for the current device (queried once, thread-safe via static init).
inline int get_num_sms() {
    static int num_sms = [] {
        int device;
        cudaGetDevice(&device);
        int sms;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        return sms;
    }();
    return num_sms;
}

// Compute persistent grid size: min(num_items, occupancy * num_sms).
// KernelFunc must be a __global__ function pointer.
template <typename KernelFunc>
int persistent_grid_size(KernelFunc kern, int block_size, int smem_bytes, int num_items) {
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kern, block_size, smem_bytes);
    if (blocks_per_sm <= 0) blocks_per_sm = 1;
    return std::min(num_items, blocks_per_sm * get_num_sms());
}

} // namespace bsptc
