#pragma once

#include <tuple>
#include <cstdint>

namespace bsptc {

// Packet: descriptor + metadata components + source offsets for pointer patching.
// Components are attached by analyzer passes and carried through the pipeline.
template <typename Desc, typename... Components>
struct Packet {
    Desc desc;
    std::tuple<Components...> meta;
    int64_t offset_A = 0;
    int64_t offset_B = 0;

    template <typename T>
    T& get() { return std::get<T>(meta); }

    template <typename T>
    const T& get() const { return std::get<T>(meta); }
};

// Type-level append: Packet<D, A, B> + C -> Packet<D, A, B, C>
namespace detail {
template <typename Pkt, typename C>
struct PacketAppendImpl;

template <typename Desc, typename... Cs, typename C>
struct PacketAppendImpl<Packet<Desc, Cs...>, C> {
    using type = Packet<Desc, Cs..., C>;
};
} // namespace detail

template <typename Pkt, typename C>
using PacketWith = typename detail::PacketAppendImpl<Pkt, C>::type;

// Extract desc type from a Packet
template <typename Pkt>
struct DescOfImpl;

template <typename Desc, typename... Cs>
struct DescOfImpl<Packet<Desc, Cs...>> {
    using type = Desc;
};

template <typename Pkt>
using DescOf = typename DescOfImpl<Pkt>::type;

} // namespace bsptc
