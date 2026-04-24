#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>

namespace bsptc {

// BatchExpander: transformer that expands a desc with batch_size > 1
// into batch_size individual descs with adjusted A/B/C pointers and batch_size=1.
//
// Requires the descriptor to have:
//   A, B, C          — void* pointers
//   batch_size       — number of batch slices
//   batch_stride_a/b/c — element strides per batch slice
//
// elem_size is sizeof(T), needed for void* pointer arithmetic.
struct BatchExpander {
    size_t elem_size;

    template <typename Pkt, typename Next>
    void operator()(const Pkt& p, Next& next) const {
        const auto& sd = p.desc;
        if (sd.batch_size <= 1) {
            next.push(p);
            return;
        }

        const int64_t esz = static_cast<int64_t>(elem_size);
        const char* base_A = static_cast<const char*>(sd.A);
        const char* base_B = static_cast<const char*>(sd.B);
        char* base_C = static_cast<char*>(sd.C);

        for (int64_t b = 0; b < sd.batch_size; ++b) {
            Pkt out = p;
            out.desc.A = base_A + b * sd.batch_stride_a * esz;
            out.desc.B = base_B + b * sd.batch_stride_b * esz;
            out.desc.C = base_C + b * sd.batch_stride_c * esz;
            out.desc.batch_size = 1;
            out.desc.batch_stride_a = 0;
            out.desc.batch_stride_b = 0;
            out.desc.batch_stride_c = 0;
            next.push(std::move(out));
        }
    }
};

} // namespace bsptc
