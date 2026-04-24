#pragma once

#include "types.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace bsptc {

// Host-side data container that owns a byte buffer and exposes typed views.
class HostDataHandler {
public:
    HostDataHandler() = default;

    HostDataHandler(std::size_t elements, DataType dtype_)
        : dtype(dtype_) {
        allocate(elements);
    }
    
    void set_data_type(DataType dtype_) {
        dtype = dtype_;
    }
    
    void allocate(std::size_t elements) {
        num_elements = elements;
        buffer.resize(elements * sizeof_dtype(dtype));
    }

    void clear() {
        buffer.clear();
        num_elements = 0;
    }

    std::size_t elements() const { return num_elements; }
    DataType data_type() const { return dtype; }

    void* raw_data() { return buffer.data(); }
    const void* raw_data() const { return buffer.data(); }

    // -------------------------
    // Typed access (unchecked)
    // -------------------------

    template <typename T>
    T* data_as() {
        return reinterpret_cast<T*>(buffer.data());
    }

    template <typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(buffer.data());
    }

    // -------------------------
    // Typed setter with conversion
    // -------------------------
    // Need to add half type and change the way elements are set if src is of type float or double
    // Pseudocode: validate element count, switch on dtype, static_cast each src value into buffer.
    template <typename T>
    void set_from(const T* src, std::size_t count) {
        if (count != num_elements) {
            throw std::runtime_error("HostDataHandler: element count mismatch");
        }

        switch (dtype) {
            case DataType::FLOAT32: {
                float* dst = data_as<float>();
                for (std::size_t i = 0; i < count; ++i)
                    dst[i] = static_cast<float>(src[i]);
                break;
            }
            case DataType::FLOAT64: {
                double* dst = data_as<double>();
                for (std::size_t i = 0; i < count; ++i)
                    dst[i] = static_cast<double>(src[i]);
                break;
            }
            case DataType::FLOAT16:
                throw std::runtime_error("FLOAT16 conversion not implemented (by design)");
            default:
                throw std::runtime_error("HostDataHandler: unsupported data type");
        }
    }

private:
    DataType dtype = DataType::FLOAT32;
    std::size_t num_elements = 0;
    std::vector<std::uint8_t> buffer;
};

} // namespace bsptc
