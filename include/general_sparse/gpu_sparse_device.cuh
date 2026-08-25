#pragma once

#include "gpu_sparse_input.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

// Device-side data structures for GPU sparse tensor operations.
namespace bsptc {

// Device pointers for GPU sparse input (move-only, frees GPU memory on destruction
// unless borrowing caller-owned storage — see borrow_device_input).
// Only d_values is uploaded; block metadata stays on host.
template <typename T>
struct DeviceSparseInput {
    T* d_values = nullptr;
    bool owns = true;

    dim_t rank = 0;
    dim_t num_blocks = 0;
    int64_t total_elements = 0;

    DeviceSparseInput() = default;
    ~DeviceSparseInput() {
        if (owns && d_values) cudaFree(d_values);
    }
    DeviceSparseInput(DeviceSparseInput&& o) noexcept
        : d_values(o.d_values), owns(o.owns),
          rank(o.rank), num_blocks(o.num_blocks), total_elements(o.total_elements)
    {
        o.d_values = nullptr;
    }
    DeviceSparseInput& operator=(DeviceSparseInput&& o) noexcept {
        if (this != &o) { this->~DeviceSparseInput(); new(this) DeviceSparseInput(std::move(o)); }
        return *this;
    }
    DeviceSparseInput(const DeviceSparseInput&) = delete;
    DeviceSparseInput& operator=(const DeviceSparseInput&) = delete;
};

// Wrap caller-owned device storage (e.g. a Julia CuArray) without taking
// ownership — the destructor will not free d_values.
template <typename T>
DeviceSparseInput<T> borrow_device_input(T* d_values, dim_t rank,
                                         dim_t num_blocks, int64_t total_elements) {
    DeviceSparseInput<T> dev;
    dev.d_values = d_values;
    dev.owns = false;
    dev.rank = rank;
    dev.num_blocks = num_blocks;
    dev.total_elements = total_elements;
    return dev;
}

// Block pair info for matched blocks
struct BlockPairInfo {
    int32_t block_idx_A;    // index into A's blocks
    int32_t block_idx_B;    // index into B's blocks
    int64_t output_offset;  // offset into output buffer
    int64_t output_volume;  // size of output block
};

// Check CUDA errors
inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Copy GPUSparseInput to device (values only — metadata stays on host)
template <typename T>
DeviceSparseInput<T> copy_sparse_to_device(const GPUSparseInput<T>& host) {
    DeviceSparseInput<T> dev;
    dev.rank = host.rank;
    dev.num_blocks = host.num_blocks;
    dev.total_elements = static_cast<int64_t>(host.get_values_count());

    // Allocate and copy values (skip if empty - metadata-only mode)
    const std::size_t values_count = host.get_values_count();
    if (values_count > 0) {
        const std::size_t values_bytes = values_count * sizeof(T);
        check_cuda(cudaMalloc(&dev.d_values, values_bytes), "malloc values");
        check_cuda(cudaMemcpy(dev.d_values, host.get_values_data(),
                              values_bytes, cudaMemcpyHostToDevice), "memcpy values");
    } else {
        dev.d_values = nullptr;
    }

    return dev;
}

} // namespace bsptc
