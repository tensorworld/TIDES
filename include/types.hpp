#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>

// Fundamental enums and helpers used across the block-sparse codebase.
namespace bsptc {

// Scalar type for per-block dimension sizes (individual dims, M/N/K, strides).
// Products/offsets that can exceed 2^31 remain int64_t.
using dim_t = std::int32_t;

// Constants
inline constexpr std::size_t MAX_ELEMENTS_TO_PRINT = 64;


// -------------------------
// Fundamental enums
// -------------------------

enum class DataType : std::uint8_t {
    FLOAT16,   // half precision
    FLOAT32,   // single precision
    FLOAT64    // double precision
};

enum class LayoutType : std::uint8_t {
    RowMajor,         // rightmost dim contiguous
    ColumnMajor,      // leftmost dim contiguous
    LeftMajorStride,  // explicit column-major stride variant
    RightMajorStride  // explicit row-major stride variant
};

enum class PropType : std::uint8_t {
    DEFAULT,
    AMPERE_F64_SMALL,
    AMPERE_F64_MED,
    AMPERE_F64_LARGE,
    AMPERE_F32_SMALL,
    AMPERE_F32_MED,
    AMPERE_F32_LARGE,
    AMPERE_F16_SMALL,
    AMPERE_F16_MED,
    AMPERE_F16_LARGE,
    HOPPER_F64_SMALL,
    HOPPER_F64_MED,
    HOPPER_F64_LARGE,
    HOPPER_F32_SMALL,
    HOPPER_F32_MED,
    HOPPER_F32_LARGE,
    HOPPER_F16_SMALL,
    HOPPER_F16_MED,
    HOPPER_F16_LARGE
};

enum class Impl : std::uint8_t {
    GROUPED, // grouped GEMM path
    SINGLE   // per-problem GEMM path
};
// -------------------------
// Type utilities
// -------------------------

inline std::size_t sizeof_dtype(DataType dt) {
    switch (dt) {
        case DataType::FLOAT16: return 2;
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        default: throw std::runtime_error("Unknown data type");
    }
}

} // namespace bsptc
