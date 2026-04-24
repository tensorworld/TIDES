#pragma once

#include "tensor_meta.hpp"
#include "host_data_handler.hpp"
#include "layout_utils.hpp"
#include "types.hpp"

#include <numeric>
#include <vector>

namespace bsptc {

// Simple host-only dense tensor; stores shape, layout/strides and host buffer.
class DenseTensor {
public:
    TensorMeta meta;
    std::vector<std::int64_t> shape;
    LayoutType layout = LayoutType::RowMajor;
    std::vector<std::int64_t> strides;
    HostDataHandler host_data;   // HOST ONLY

    DenseTensor() = default;

    DenseTensor(const TensorMeta& meta_,
                const std::vector<std::int64_t>& shape_,
                LayoutType layout_)
        : meta(meta_),
          shape(shape_),
          layout(layout_),
          strides(layout::compute_strides(shape_, layout_)),
          host_data(numel_from_shape(shape_), meta_.dtype) {}

    ~DenseTensor() = default;

    std::size_t numel() const {
        return numel_from_shape(shape);
    }

    void* data() { return host_data.raw_data(); }
    const void* data() const { return host_data.raw_data(); }

private:
    static std::size_t numel_from_shape(const std::vector<std::int64_t>& shape) {
        return std::accumulate(
            shape.begin(), shape.end(),
            static_cast<std::size_t>(1),
            std::multiplies<std::size_t>());
    }
};

} // namespace bsptc
