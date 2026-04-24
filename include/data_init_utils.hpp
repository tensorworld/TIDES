#pragma once

#include "block_sparse_tensor.hpp"
#include "block_utils.hpp"
#include "dense_tensor.hpp"
#include "layout_utils.hpp"
#include "types.hpp"

#include <vector>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Host-side helpers for initializing and reordering tensor storage.
// Reorder/transpose routines copy into a temp buffer using computed strides.
namespace bsptc {
namespace utils {

// ============================================================
// Sequential initialization
// ============================================================

template <typename T>
void init_sequential(DenseTensor& tensor, T start = T{0}) {
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 sequential init not supported on host");
    }

    T* ptr = tensor.host_data.data_as<T>();
    std::size_t n = tensor.numel();

    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<T>(start + static_cast<T>(i));
    }
}

template <typename T>
void init_sequential(BlockSparseTensor& tensor, T start = T{0}) {
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 sequential init not supported on host");
    }

    T* ptr = tensor.host_data.data_as<T>();
    auto offsets = block_utils::compute_block_offsets(tensor);

    std::size_t block_idx = 0;
    for (const auto& coord : tensor.block_coords) {
        const std::size_t block_volume = block_utils::block_volume_for_coord(tensor, coord);
        const std::size_t offset = offsets[block_idx++];
        for (std::size_t i = 0; i < block_volume; ++i) {
            ptr[offset + i] = static_cast<T>(start + static_cast<T>(i));
        }
    }
}

// ============================================================
// Fast xorshift32 RNG — produces floats in [0,1) with no transcendentals
// ============================================================

struct Xorshift32 {
    uint32_t state;
    explicit Xorshift32(uint32_t seed) : state(seed ? seed : 1u) {}
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    // Map to [0, 1) by dividing by 2^32
    float nextf() {
        return static_cast<float>(next()) * (1.0f / 4294967296.0f);
    }
};

// ============================================================
// Random initialization
// ============================================================

template <typename T>
void init_random(DenseTensor& tensor,
                 T low = T{0},
                 T high = T{1},
                 std::uint64_t seed = 42) {
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 random init not supported on host");
    }

    Xorshift32 rng(static_cast<uint32_t>(seed));
    T range = high - low;
    T* ptr = tensor.host_data.data_as<T>();
    std::size_t n = tensor.numel();

    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = low + static_cast<T>(rng.nextf()) * range;
    }
}

template <typename T>
void init_random(BlockSparseTensor& tensor,
                 T low = T{0},
                 T high = T{1},
                 std::uint64_t seed = 42) {
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 random init not supported on host");
    }

    T* ptr = tensor.host_data.data_as<T>();
    T range = high - low;
    const int64_t total = static_cast<int64_t>(block_utils::total_elements(tensor));

#ifdef _OPENMP
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
#else
    {
        int tid = 0;
        int nthreads = 1;
#endif
        int64_t chunk = (total + nthreads - 1) / nthreads;
        int64_t start = tid * chunk;
        int64_t end = std::min(start + chunk, total);
        Xorshift32 rng(static_cast<uint32_t>(seed ^ (static_cast<uint64_t>(tid) * 2654435761ULL)));
        for (int64_t i = start; i < end; ++i) {
            ptr[i] = low + static_cast<T>(rng.nextf()) * range;
        }
    }
}
// ============================================================
// Layout-based reorder / transpose
// ============================================================
//
// This performs a FULL layout transform:
//   source strides  -> destination strides
//
// Example:
//   RowMajor -> ColumnMajor
//   RightMajorStride -> LeftMajorStride
//
// ============================================================

template <typename T>
void reorder_by_layout(DenseTensor& tensor,
                       LayoutType src_layout,
                       LayoutType dst_layout) {
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 reorder not supported on host");
    }

    const auto& shape = tensor.shape;
    std::size_t ndim = shape.size();
    std::size_t total = tensor.numel();

    // Compute strides
    auto src_strides = layout::compute_strides(shape, src_layout);
    auto dst_strides = layout::compute_strides(shape, dst_layout);

    // Temporary buffer
    std::vector<T> tmp(total);

    T* src = tensor.host_data.data_as<T>();

    // Multi-index counter
    std::vector<std::int64_t> idx(ndim, 0);

    for (std::size_t linear = 0; linear < total; ++linear) {

        // Convert linear index -> multi-index (row-major traversal)
        std::size_t r = linear;
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            idx[d] = r % shape[d];
            r /= shape[d];
            if (d == 0) break;
        }

        std::size_t src_off = layout::linear_index(idx, src_strides);
        std::size_t dst_off = layout::linear_index(idx, dst_strides);

        tmp[dst_off] = src[src_off];
    }

    // Copy back
    std::memcpy(src, tmp.data(), total * sizeof(T));

    // Update tensor layout + strides
    tensor.strides = dst_strides;
    tensor.layout = dst_layout;
}

template <typename T>
void reorder_by_layout(BlockSparseTensor& tensor,
                       LayoutType src_layout,
                       LayoutType dst_layout) {
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 reorder not supported on host");
    }

    if (tensor.block_coords.empty()) {
        tensor.strides.clear();
        tensor.layout = dst_layout;
        return;
    }

    auto offsets = block_utils::compute_block_offsets(tensor);
    T* data = tensor.host_data.data_as<T>();

    std::size_t b = 0;
    for (const auto& coord : tensor.block_coords) {
        const auto block_shape = block_utils::block_shape_for_coord(tensor, coord);
        const auto block_volume = block_utils::block_volume_for_coord(tensor, coord);

        auto src_strides = layout::compute_strides(block_shape, src_layout);
        auto dst_strides = layout::compute_strides(block_shape, dst_layout);

        std::vector<T> tmp(block_volume);
        std::vector<std::int64_t> idx(block_shape.size(), 0);

        T* block = data + offsets[b++];

        for (std::size_t linear = 0; linear < block_volume; ++linear) {

            // Convert linear index -> multi-index (row-major traversal)
            std::size_t r = linear;
            for (std::int64_t d = static_cast<std::int64_t>(block_shape.size()) - 1; d >= 0; --d) {
                idx[d] = static_cast<std::int64_t>(r % static_cast<std::size_t>(block_shape[d]));
                r /= static_cast<std::size_t>(block_shape[d]);
                if (d == 0) break;
            }

            std::size_t src_off = layout::linear_index(idx, src_strides);
            std::size_t dst_off = layout::linear_index(idx, dst_strides);

            tmp[dst_off] = block[src_off];
        }

        // Copy back
        std::memcpy(block, tmp.data(), block_volume * sizeof(T));
    }
    // Update tensor layout + strides using the first block as reference
    tensor.layout = dst_layout;
    tensor.strides =
        layout::compute_strides(
            block_utils::block_shape_for_coord(tensor, *tensor.block_coords.begin()),
            dst_layout);
}
// ============================================================
// Explicit transpose (2D only)
// ============================================================

template <typename T>
void transpose_2d(DenseTensor& tensor) {
    if (tensor.meta.ndim != 2) {
        throw std::runtime_error("transpose_2d requires rank-2 tensor");
    }
    if (tensor.meta.dtype == DataType::FLOAT16) {
        throw std::runtime_error("FLOAT16 transpose not supported on host");
    }

    const auto& shape = tensor.shape;
    std::size_t rows = shape[0];
    std::size_t cols = shape[1];

    T* src = tensor.host_data.data_as<T>();
    std::vector<T> tmp(rows * cols);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            tmp[j * rows + i] = src[i * cols + j];
        }
    }

    // Swap shape
    tensor.shape = { static_cast<std::int64_t>(cols),
                          static_cast<std::int64_t>(rows) };

    // Copy data back
    std::memcpy(src, tmp.data(), rows * cols * sizeof(T));

    // Recompute strides (row-major default after transpose)
    tensor.strides =
        layout::compute_strides(tensor.shape,
                                LayoutType::RowMajor);
    tensor.layout = LayoutType::RowMajor;
}

} // namespace utils
} // namespace bsptc
