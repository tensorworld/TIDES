#pragma once

#include "types.hpp"

#include <cutensor.h>
#include <stdexcept>

// Shared cuTENSOR utility functions used by block-sparse contraction code.
namespace bsptc {

inline cudaDataType_t to_cutensor_type(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return CUDA_R_32F;
        case DataType::FLOAT64: return CUDA_R_64F;
        default: throw std::runtime_error("cutensor: unsupported dtype");
    }
}

inline cutensorComputeDescriptor_t to_cutensor_compute(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return CUTENSOR_COMPUTE_DESC_32F;
        case DataType::FLOAT64: return CUTENSOR_COMPUTE_DESC_64F;
        default: throw std::runtime_error("cutensor: unsupported dtype");
    }
}

inline cutensorHandle_t get_cached_cutensor_handle() {
    static cutensorHandle_t h = []() {
        cutensorHandle_t handle{};
        if (cutensorCreate(&handle) != CUTENSOR_STATUS_SUCCESS)
            throw std::runtime_error("cutensorCreate failed");
        return handle;
    }();
    return h;
}

} // namespace bsptc
