#pragma once

#include "tensor_meta.hpp"
#include "host_data_handler.hpp"
#include "types.hpp"
#include <vector>
#include <string>
#include <map>
#include <set>

// Host-side representation of a block-sparse tensor (metadata + coords + host buffer).
namespace bsptc {

// Block-sparse tensor (host representation)
class BlockSparseTensor {
public:
    TensorMeta meta;
    LayoutType layout = LayoutType::RowMajor;                    // intra-block layout
    std::vector<std::int64_t> strides = {};                      // block strides (same for all blocks)
    std::vector<int> num_blocks;                                 // number of blocks per dimension
    std::map<char, std::vector<std::int64_t>> block_sizes;       // per-index block sizes
    std::set<std::vector<std::int64_t>> block_coords;            // coordinates of stored blocks

    // Host-side storage of all blocks
    HostDataHandler host_data;

    BlockSparseTensor() = default;

    BlockSparseTensor(const TensorMeta& meta_,
                      const std::vector<int> num_blocks_, 
                      std::map<char, std::vector<std::int64_t>> block_sizes_,
                      std::set<std::vector<std::int64_t>> block_coords_,
                      LayoutType layout_ = LayoutType::RowMajor)
        : meta(meta_),
          layout(layout_),
          num_blocks(num_blocks_),
          block_sizes(block_sizes_),
          block_coords(block_coords_)          
    {
        host_data.set_data_type(meta.dtype);
    }

    void* data() { return host_data.raw_data(); }
    const void* data() const { return host_data.raw_data(); }
    };

} // namespace bsptc
