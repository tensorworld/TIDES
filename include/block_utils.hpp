#pragma once

#include "block_sparse_tensor.hpp"

#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace bsptc {
// Derived shapes/offset helpers for block-sparse tensors.
// Example: indices ['a','b'], block_sizes a:{2,3}, b:{4,5} -> dense shape [5,9].
// compute_block_offsets walks block_coords in order, accumulating element counts.
namespace block_utils {

// Compute the dense shape by summing block sizes along each index
inline std::vector<std::int64_t>
compute_full_shape(const BlockSparseTensor& tensor) {
    std::vector<std::int64_t> shape;
    shape.reserve(tensor.meta.indices.size());

    for (char idx : tensor.meta.indices) {
        const auto it = tensor.block_sizes.find(idx);
        if (it == tensor.block_sizes.end()) {
            throw std::runtime_error("Missing block size entry for index " + std::string(1, idx));
        }
        shape.push_back(std::accumulate(
            it->second.begin(), it->second.end(),
            static_cast<std::int64_t>(0)));
    }
    return shape;
}

// Block shape for a specific block coordinate
inline std::vector<std::int64_t>
block_shape_for_coord(const BlockSparseTensor& tensor,
                      const std::vector<std::int64_t>& coord) {
    if (coord.size() != tensor.meta.indices.size()) {
        throw std::runtime_error("block_shape_for_coord: rank mismatch");
    }

    std::vector<std::int64_t> shape;
    shape.reserve(coord.size());
    for (std::size_t dim = 0; dim < coord.size(); ++dim) {
        char label = tensor.meta.indices[dim];
        auto sizes_it = tensor.block_sizes.find(label);
        if (sizes_it == tensor.block_sizes.end()) {
            throw std::runtime_error("Missing block size for index " + std::string(1, label));
        }
        const auto section = coord[dim];
        if (section < 0 || section >= static_cast<std::int64_t>(sizes_it->second.size())) {
            throw std::runtime_error("Block coordinate out of range for index " + std::string(1, label));
        }
        shape.push_back(sizes_it->second[static_cast<std::size_t>(section)]);
    }
    return shape;
}

inline std::size_t
block_volume_for_coord(const BlockSparseTensor& tensor,
                       const std::vector<std::int64_t>& coord) {
    auto shape = block_shape_for_coord(tensor, coord);
    return std::accumulate(
        shape.begin(), shape.end(),
        static_cast<std::size_t>(1),
        std::multiplies<std::size_t>());
}

// Offsets (in elements) for each stored block, following the order of block_coords iteration
inline std::vector<std::size_t>
compute_block_offsets(const BlockSparseTensor& tensor) {
    std::vector<std::size_t> offsets;
    offsets.reserve(tensor.block_coords.size());

    std::size_t offset = 0;
    for (const auto& coord : tensor.block_coords) {
        offsets.push_back(offset);
        offset += block_volume_for_coord(tensor, coord);
    }
    return offsets;
}

inline std::size_t
total_elements(const BlockSparseTensor& tensor) {
    std::size_t total = 0;
    for (const auto& coord : tensor.block_coords) {
        total += block_volume_for_coord(tensor, coord);
    }
    return total;
}

} // namespace block_utils
} // namespace bsptc
