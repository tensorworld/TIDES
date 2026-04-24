#pragma once

#include "../../pipeline_descs.hpp"

#include <cstdint>
#include <cstddef>
#include <utility>

namespace bsptc {

// KSplitter: transformer that splits the K dimension into chunks.
//
// For each packet with K > k_threshold, emits ceil(K / k_threshold) sub-packets
// with adjusted K and advanced A/B pointers. C pointer is unchanged — all chunks
// accumulate to the same output location via atomicAdd in the kernel.
//
// Packets with K <= k_threshold pass through unchanged.
struct KSplitter {
    dim_t k_threshold;
    bool a_k_inner;   // NN or NT: A is row-major (K is inner dim)
    bool b_n_inner;   // NN or TN: B is row-major (N is inner dim)
    size_t elem_size; // sizeof(T) for pointer arithmetic on void*

    template <typename Pkt, typename Next>
    void operator()(const Pkt& p, Next& next) const {
        using DescT = std::decay_t<decltype(p.desc)>;
        static_assert(std::is_base_of_v<F_Strides, DescT>,
                      "KSplitter requires a descriptor with F_Strides (lda, ldb, ldc)");
        const auto& sd = p.desc;

        if (sd.K <= k_threshold) {
            next.push(p);
            return;
        }

        const int64_t esz = static_cast<int64_t>(elem_size);

        // A stride per K-step: row-major (a_k_inner) → 1 element per K step;
        // column-major → lda elements per K step.
        const int64_t a_k_stride = a_k_inner
            ? static_cast<int64_t>(1) : static_cast<int64_t>(sd.lda);
        // B stride per K-step: row-major (b_n_inner) → ldb elements per K step;
        // column-major → 1 element per K step.
        const int64_t b_k_stride = b_n_inner
            ? static_cast<int64_t>(sd.ldb) : static_cast<int64_t>(1);

        const char* base_A = static_cast<const char*>(sd.A);
        const char* base_B = static_cast<const char*>(sd.B);

        dim_t k_off = 0;
        while (k_off < sd.K) {
            dim_t k_chunk = (sd.K - k_off >= k_threshold)
                ? k_threshold : (sd.K - k_off);

            Pkt out = p;
            out.desc.A = base_A + k_off * a_k_stride * esz;
            out.desc.B = base_B + k_off * b_k_stride * esz;
            // C unchanged — all K-chunks accumulate to same output
            out.desc.K = k_chunk;
            // lda, ldb, ldc unchanged (stride within the matrix is the same)
            next.push(std::move(out));

            k_off += k_chunk;
        }
    }
};

} // namespace bsptc
