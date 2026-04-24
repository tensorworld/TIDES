#pragma once

#include "../../pipeline_descs.hpp"

#include <cstdint>
#include <cstddef>
#include <utility>

namespace bsptc {

// MNSplitter: transformer that tiles M and N into threshold-aligned chunks.
//
// M=1500, threshold=512 → chunks [512, 512, 476]
// M=400,  threshold=512 → no split
//
// Each M-chunk is further tiled along N the same way.
// Produces ceil(M/threshold) * ceil(N/threshold) packets per input.
struct MNSplitter {
    dim_t threshold;
    bool a_k_inner;   // NN or NT: A is row-major (K is inner dim)
    bool b_n_inner;   // NN or TN: B is row-major (N is inner dim)
    size_t elem_size; // sizeof(T) for pointer arithmetic on void*

    template <typename Pkt, typename Next>
    void operator()(const Pkt& p, Next& next) const {
        using DescT = std::decay_t<decltype(p.desc)>;
        static_assert(std::is_base_of_v<F_Strides, DescT>,
                      "MNSplitter requires a descriptor with F_Strides (lda, ldb, ldc)");
        const auto& sd = p.desc;
        const int64_t esz = static_cast<int64_t>(elem_size);

        // A stride per M-row: row-major (a_k_inner) → lda elements per row;
        // column-major → 1 element per row (contiguous in memory).
        const int64_t a_m_stride = a_k_inner
            ? static_cast<int64_t>(sd.lda) : static_cast<int64_t>(1);
        // B stride per N-column: row-major (b_n_inner) → 1 element per col;
        // column-major → ldb elements per col.
        const int64_t b_n_stride = b_n_inner
            ? static_cast<int64_t>(1) : static_cast<int64_t>(sd.ldb);
        // C stride: row-major, N is inner → 1 per col, ldc per row
        const int64_t c_m_stride = static_cast<int64_t>(sd.ldc);
        const int64_t c_n_stride = static_cast<int64_t>(1);

        const char* base_A = static_cast<const char*>(sd.A);
        const char* base_B = static_cast<const char*>(sd.B);
        char* base_C = static_cast<char*>(sd.C);

        dim_t m_off = 0;
        while (m_off < sd.M) {
            dim_t m_chunk = (sd.M - m_off >= threshold) ? threshold : (sd.M - m_off);

            dim_t n_off = 0;
            while (n_off < sd.N) {
                dim_t n_chunk = (sd.N - n_off >= threshold) ? threshold : (sd.N - n_off);

                Pkt out = p;
                out.desc.A = base_A + m_off * a_m_stride * esz;
                out.desc.B = base_B + n_off * b_n_stride * esz;
                out.desc.C = base_C + (m_off * c_m_stride + n_off * c_n_stride) * esz;
                out.desc.M = m_chunk;
                out.desc.N = n_chunk;
                out.desc.K = sd.K;
                out.desc.lda = sd.lda;
                out.desc.ldb = sd.ldb;
                out.desc.ldc = sd.ldc;
                next.push(std::move(out));

                n_off += n_chunk;
            }
            m_off += m_chunk;
        }
    }
};

} // namespace bsptc
