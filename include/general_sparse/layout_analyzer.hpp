#pragma once

#include "cute_layout_analysis.hpp"
#include "gpu_sparse_input.hpp"

#include <algorithm>
#include <numeric>

// Layout analysis engine implementing CuTe-style coalesce logic.
// Analyzes block memory layouts for scheduling optimization.
namespace bsptc {

class LayoutAnalyzer {
public:
    // Analyze a single block's layout from a GPUSparseInput tensor
    template <typename T>
    static BlockLayoutDescriptor analyze_block(
        const GPUSparseInput<T>& tensor,
        int64_t block_idx)
    {
        BlockLayoutDescriptor desc;

        // Extract block dimensions
        const dim_t rank = tensor.rank;
        desc.shape.resize(rank);
        desc.stride.resize(rank);

        for (int32_t d = 0; d < rank; ++d) {
            desc.shape[d] = tensor.block_dims[block_idx * rank + d];
        }

        // Compute row-major strides (default layout)
        compute_row_major_strides(desc.shape, desc.stride);

        // Compute volume
        desc.volume = std::accumulate(desc.shape.begin(), desc.shape.end(),
                                       int64_t{1}, std::multiplies<int64_t>());

        // Apply coalescing
        coalesce_dimensions(desc);

        // Classify the layout
        desc.layout_class = classify_layout(desc);

        // Compute properties
        desc.vector_width = compute_potential_vector_width(desc);
        desc.max_contiguous_span = compute_max_contiguous_span(desc);

        // Check memory order
        desc.is_row_major = check_row_major(desc.coalesced_shape, desc.coalesced_stride);
        desc.is_col_major = check_col_major(desc.coalesced_shape, desc.coalesced_stride);

        return desc;
    }

private:
    // Compute row-major strides from shape
    static void compute_row_major_strides(const std::vector<int64_t>& shape,
                                           std::vector<int64_t>& stride) {
        const size_t rank = shape.size();
        stride.resize(rank);
        if (rank == 0) return;

        stride[rank - 1] = 1;
        for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
            stride[d] = stride[d + 1] * shape[d + 1];
        }
    }

    // Implements CuTe's bw_coalesce logic for runtime values
    // Merge condition: stride[i] == shape[i+1] * stride[i+1]
    // Process right-to-left (row-major order)
    static void coalesce_dimensions(BlockLayoutDescriptor& desc) {
        const size_t rank = desc.shape.size();
        if (rank == 0) {
            desc.coalesced_shape = {1};
            desc.coalesced_stride = {0};
            return;
        }

        std::vector<int64_t> coal_shape;
        std::vector<int64_t> coal_stride;

        // Start with the rightmost dimension (fastest-varying in row-major)
        int64_t current_shape = desc.shape[rank - 1];
        int64_t current_stride = desc.stride[rank - 1];

        // Process from second-to-last to first (right-to-left)
        for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
            int64_t next_shape = desc.shape[d];
            int64_t next_stride = desc.stride[d];

            // Skip size-1 dimensions
            if (next_shape == 1) {
                continue;
            }

            // Check merge condition: next_stride == current_shape * current_stride
            if (next_stride == current_shape * current_stride) {
                // Merge: extend current dimension
                current_shape = next_shape * current_shape;
                // Stride stays as current_stride (the smaller/inner stride)
            } else {
                // Can't merge: save current and start new
                if (current_shape != 1) {
                    coal_shape.push_back(current_shape);
                    coal_stride.push_back(current_stride);
                }
                current_shape = next_shape;
                current_stride = next_stride;
            }
        }

        // Don't forget the last accumulated dimension
        if (current_shape != 1) {
            coal_shape.push_back(current_shape);
            coal_stride.push_back(current_stride);
        }

        // Handle empty result (all dimensions were 1)
        if (coal_shape.empty()) {
            coal_shape = {1};
            coal_stride = {0};
        }

        // Reverse to restore left-to-right order (slowest to fastest)
        std::reverse(coal_shape.begin(), coal_shape.end());
        std::reverse(coal_stride.begin(), coal_stride.end());

        desc.coalesced_shape = std::move(coal_shape);
        desc.coalesced_stride = std::move(coal_stride);
    }

    // Classify layout based on coalesced result
    static LayoutClass classify_layout(const BlockLayoutDescriptor& desc) {
        const size_t rank = desc.coalesced_shape.size();

        // Single dimension after coalescing = fully contiguous
        if (rank == 1) {
            // Check if stride is 1 (true contiguous) or 0 (broadcast)
            if (desc.coalesced_stride[0] <= 1) {
                return LayoutClass::Contiguous;
            }
        }

        // Check if layout is contiguous (covers both row-major and column-major)
        if (is_contiguous_layout(desc.coalesced_shape, desc.coalesced_stride)) {
            return LayoutClass::Contiguous;
        }

        // Check for strided pattern: regular strides with gaps
        bool is_regular = true;

        // Sort dimensions by stride to check regularity
        std::vector<size_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return desc.coalesced_stride[a] < desc.coalesced_stride[b];
        });

        int64_t expected = 1;
        for (size_t i : order) {
            int64_t actual = desc.coalesced_stride[i];
            if (actual < expected) {
                is_regular = false;
                break;
            }
            if (expected > 0 && actual % expected != 0) {
                is_regular = false;
                break;
            }
            expected = actual * desc.coalesced_shape[i];
        }

        if (is_regular) {
            return LayoutClass::Strided;
        } else {
            return LayoutClass::Scattered;
        }
    }

    // Check if a layout is contiguous (elements form a contiguous range)
    static bool is_contiguous_layout(const std::vector<int64_t>& shape,
                                      const std::vector<int64_t>& stride) {
        const size_t rank = shape.size();
        if (rank == 0) return true;
        if (rank == 1) return stride[0] <= 1;

        // Sort dimensions by stride
        std::vector<size_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return stride[a] < stride[b];
        });

        // Check contiguity: stride[order[i]] == product of shapes[order[0..i-1]]
        int64_t expected = 1;
        for (size_t i = 0; i < rank; ++i) {
            size_t dim = order[i];
            if (shape[dim] == 1) {
                continue;
            }
            if (stride[dim] != expected) {
                return false;
            }
            expected *= shape[dim];
        }

        return true;
    }

    // Convenience overload that returns potential vector width (shape-based only)
    static int32_t compute_potential_vector_width(const BlockLayoutDescriptor& desc) {
        if (desc.coalesced_shape.empty()) {
            return 1;
        }

        int64_t inner_stride = desc.coalesced_stride.back();
        if (inner_stride != 1) {
            return 1;
        }

        int64_t inner_size = desc.coalesced_shape.back();

        for (int32_t w : {16, 8, 4, 2}) {
            if (inner_size >= w && inner_size % w == 0) {
                return w;
            }
        }

        return 1;
    }

    // Compute maximum contiguous span in elements
    static int64_t compute_max_contiguous_span(const BlockLayoutDescriptor& desc) {
        if (desc.coalesced_shape.empty()) {
            return 1;
        }

        // For contiguous layout, entire volume is contiguous
        if (desc.layout_class == LayoutClass::Contiguous) {
            return desc.volume;
        }

        // For strided, find the innermost contiguous run
        int64_t span = 1;
        for (int d = static_cast<int>(desc.coalesced_shape.size()) - 1; d >= 0; --d) {
            if (desc.coalesced_stride[d] == span) {
                span *= desc.coalesced_shape[d];
            } else {
                break;
            }
        }

        return span;
    }

    // Check if layout is row-major
    static bool check_row_major(const std::vector<int64_t>& shape,
                                 const std::vector<int64_t>& stride) {
        if (shape.empty()) return true;

        int64_t expected = 1;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            if (stride[d] != expected && shape[d] > 1) {
                return false;
            }
            expected *= shape[d];
        }
        return true;
    }

    // Check if layout is column-major
    static bool check_col_major(const std::vector<int64_t>& shape,
                                 const std::vector<int64_t>& stride) {
        if (shape.empty()) return true;

        int64_t expected = 1;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (stride[d] != expected && shape[d] > 1) {
                return false;
            }
            expected *= shape[d];
        }
        return true;
    }
};

} // namespace bsptc
