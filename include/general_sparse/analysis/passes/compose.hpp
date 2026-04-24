#pragma once

#include <utility>

namespace bsptc {

// Compose<A, B>: chains two transformers A and B.
// A expands packets (1-to-N), B is applied to each output of A.
// Usage: .transform(compose(MNSplitter{...}, KSplitter{...}))
template <typename A, typename B>
struct Compose {
    A a;
    B b;

    template <typename Pkt, typename Next>
    void operator()(const Pkt& p, Next& next) const {
        // Bridge connects A's output to B, which then pushes to Next.
        struct Bridge {
            const B& b_ref;
            Next& next_ref;
            void push(Pkt pkt) { b_ref(pkt, next_ref); }
        };
        Bridge bridge{b, next};
        a(p, bridge);
    }
};

template <typename A, typename B>
Compose<A, B> compose(A a, B b) {
    return {std::move(a), std::move(b)};
}

} // namespace bsptc
