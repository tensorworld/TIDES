#pragma once

#include <cstdint>
#include <vector>

// Layout analysis data structures for block sparse tensor scheduling.
// Implements CuTe-inspired layout algebra for runtime analysis.
namespace bsptc {

// Classification of memory access patterns
enum class LayoutClass : uint8_t {
    Contiguous,  // Row/Col-major, no gaps between elements
    Strided,     // Regular strides with gaps (stride > shape)
    Scattered    // Complex or irregular access pattern
};

// Full layout descriptor with detailed analysis (for host-side use)
struct BlockLayoutDescriptor {
    // Original shape and stride
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    // Coalesced shape and stride (after dimension merging)
    std::vector<int64_t> coalesced_shape;
    std::vector<int64_t> coalesced_stride;

    // Classification
    LayoutClass layout_class;

    // Computed properties
    int64_t volume;               // Total number of elements
    int64_t max_contiguous_span;  // Longest contiguous run in memory

    // Memory order flags
    bool is_row_major;
    bool is_col_major;

    // Vectorization potential (1, 2, 4, 8, 16)
    // Based on shape divisibility only; actual vectorized loads require alignment check.
    int32_t vector_width;
};

} // namespace bsptc
