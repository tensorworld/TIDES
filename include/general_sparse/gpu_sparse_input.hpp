#pragma once

#include "../block_sparse_tensor.hpp"
#include "../block_utils.hpp"
#include "../types.hpp"

#include <cstdint>
#include <vector>

// Flat, GPU-friendly representation of block sparse tensors.
// No nested containers - all arrays can be directly copied to device memory.
namespace bsptc {

// Flat GPU-friendly representation of a block sparse tensor
template <typename T>
struct GPUSparseInput {
    // Block metadata (flat arrays, contiguous)
    std::vector<int32_t> block_coords;    // [num_blocks * rank] flattened coords
    std::vector<int64_t> block_offsets;   // [num_blocks] byte offset into values
    std::vector<int64_t> block_volumes;   // [num_blocks] element count per block

    // Block dimension info (flat)
    std::vector<dim_t> block_dims;        // [num_blocks * rank] dimensions per block

    // Values — owned (vector), borrowed (pointer to external data), or absent
    // entirely (metadata-only inputs whose device values arrive via
    // borrow_device_input; total_values then records the element span).
    std::vector<T> values;                // all block data concatenated (owned)
    const T* values_ptr = nullptr;        // non-owning pointer to external data
    int64_t total_values = 0;             // element span when values is not owned

    const T* get_values_data() const { return values_ptr ? values_ptr : values.data(); }
    std::size_t get_values_count() const {
        return total_values > 0 ? static_cast<std::size_t>(total_values)
                                : values.size();
    }

    // Tensor metadata
    dim_t rank;
    dim_t num_blocks;
    std::vector<char> labels;             // index labels
    std::vector<int64_t> full_extents;    // full dense shape

    // Block sizes per dimension (for index matching)
    // block_sizes_flat: [sum of num_blocks_per_dim] flattened block sizes
    // block_sizes_offsets: [rank] offset into block_sizes_flat for each dim
    std::vector<dim_t> block_sizes_flat;
    std::vector<int32_t> block_sizes_offsets;
    std::vector<int32_t> num_blocks_per_dim;
};

// Convert BlockSparseTensor to GPU-friendly format
// Set skip_values=true for metadata-only (simulation) use
template <typename T>
GPUSparseInput<T> to_gpu_input(const BlockSparseTensor& bst, bool skip_values = false) {
    GPUSparseInput<T> out;

    out.rank = static_cast<dim_t>(bst.meta.ndim);
    out.num_blocks = static_cast<dim_t>(bst.block_coords.size());

    // Copy labels
    out.labels = bst.meta.indices;

    // Compute full extents
    out.full_extents = block_utils::compute_full_shape(bst);

    // Reserve space for flat arrays
    out.block_coords.reserve(out.num_blocks * out.rank);
    out.block_offsets.reserve(out.num_blocks);
    out.block_volumes.reserve(out.num_blocks);
    out.block_dims.reserve(out.num_blocks * out.rank);

    // Copy num_blocks_per_dim
    out.num_blocks_per_dim.reserve(bst.num_blocks.size());
    for (int nb : bst.num_blocks) {
        out.num_blocks_per_dim.push_back(static_cast<int32_t>(nb));
    }

    // Flatten block_sizes with offsets
    out.block_sizes_offsets.reserve(out.rank);
    int32_t sizes_offset = 0;
    for (int dim = 0; dim < out.rank; ++dim) {
        char label = bst.meta.indices[dim];
        const auto& sizes = bst.block_sizes.at(label);
        out.block_sizes_offsets.push_back(sizes_offset);
        for (int64_t sz : sizes) {
            out.block_sizes_flat.push_back(static_cast<dim_t>(sz));
        }
        sizes_offset += static_cast<int32_t>(sizes.size());
    }

    // Compute block offsets and flatten coordinates/dimensions
    const auto offsets = block_utils::compute_block_offsets(bst);
    std::size_t blk_idx = 0;
    for (const auto& coord : bst.block_coords) {
        // Flatten coordinates
        for (int64_t c : coord) {
            out.block_coords.push_back(static_cast<int32_t>(c));
        }

        // Store offset (in elements)
        out.block_offsets.push_back(static_cast<int64_t>(offsets[blk_idx]));

        // Compute and store volume
        auto shape = block_utils::block_shape_for_coord(bst, coord);
        int64_t vol = 1;
        for (int64_t s : shape) {
            vol *= s;
            out.block_dims.push_back(static_cast<dim_t>(s));
        }
        out.block_volumes.push_back(vol);

        ++blk_idx;
    }

    // Borrow pointer to existing host data (no copy)
    if (!skip_values) {
        out.values_ptr = bst.host_data.data_as<T>();
        out.total_values = static_cast<int64_t>(block_utils::total_elements(bst));
    }

    return out;
}

// Get the block coordinate for block index blk at dimension dim
inline int32_t get_block_coord(const int32_t* block_coords, dim_t rank,
                               int64_t blk_idx, dim_t dim) {
    return block_coords[blk_idx * rank + dim];
}

// Get the block dimension for block index blk at dimension dim
inline dim_t get_block_dim(const dim_t* block_dims, dim_t rank,
                           int64_t blk_idx, dim_t dim) {
    return block_dims[blk_idx * rank + dim];
}

} // namespace bsptc
