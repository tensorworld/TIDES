#pragma once

#include "types.hpp"
#include <string>
#include <vector>

// Basic metadata shared across dense and block-sparse tensors.
namespace bsptc {

// Metadata common for all tensor and its blocks
struct TensorMeta {
    std::string name;                      // Tensor name
    std::int16_t ndim = 0;                 // Rank
    std::vector<char> indices;             // Index labels
    DataType dtype = DataType::FLOAT32;    // Element type

    TensorMeta() = default;

    TensorMeta(std::string name_,
               std::int16_t ndim_,
               std::vector<char> indices_,
               DataType dtype_)
        : name(std::move(name_)),
          ndim(ndim_),
          indices(std::move(indices_)),
          dtype(dtype_) {}
};

} // namespace bsptc
