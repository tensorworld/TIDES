#pragma once

#include "packet.hpp"
#include <vector>
#include <utility>

namespace bsptc {

// ============================================================================
// Chain nodes (push phase)
// ============================================================================

// Terminal node: collects packets into a buffer.
template <typename Pkt>
struct Collector {
    using InputPacket = Pkt;
    using BufferType = std::vector<Pkt>;

    BufferType buffer;

    void push(Pkt p) { buffer.push_back(std::move(p)); }

    BufferType& get_buffer() { return buffer; }
    const BufferType& get_buffer() const { return buffer; }
};

// Analyzer node: enriches packet with a new component (1-to-1).
// Pass must define:
//   using Output = SomeType;
//   void operator()(const Desc& desc, Output& out) const;
template <typename Pass, typename Next>
struct AnalyzerNode {
    using OutputComponent = typename Pass::Output;
    using InputPacket = typename Next::InputPacket;
    using BufferType = typename Next::BufferType;

    Pass pass;
    Next next;

    void push(InputPacket p) {
        pass(p.desc, p.template get<OutputComponent>());
        next.push(std::move(p));
    }

    BufferType& get_buffer() { return next.get_buffer(); }
    const BufferType& get_buffer() const { return next.get_buffer(); }
};

// Transformer node: 1-to-N mapping (calls next.push() 0-N times).
// XForm must define:
//   template <typename Pkt, typename NextT>
//   void operator()(const Pkt& p, NextT& next) const;
template <typename XForm, typename Next>
struct TransformerNode {
    using InputPacket = typename Next::InputPacket;
    using BufferType = typename Next::BufferType;

    XForm xform;
    Next next;

    void push(InputPacket p) {
        xform(p, next);
    }

    BufferType& get_buffer() { return next.get_buffer(); }
    const BufferType& get_buffer() const { return next.get_buffer(); }
};

// ============================================================================
// Built pipeline: push/collect/finish interface
// ============================================================================

// Primary template: pipeline with a dispatcher (has finish())
template <typename Chain, typename Disp = void>
struct BuiltPipeline {
    using Pkt = typename Chain::InputPacket;
    using DescType = DescOf<Pkt>;

    Chain chain;
    Disp dispatcher;

    void push(DescType desc, int64_t offset_A, int64_t offset_B) {
        Pkt p;
        p.desc = std::move(desc);
        p.offset_A = offset_A;
        p.offset_B = offset_B;
        chain.push(std::move(p));
    }

    std::vector<Pkt>& collect() { return chain.get_buffer(); }
    const std::vector<Pkt>& collect() const { return chain.get_buffer(); }

    // Dispatch phase: sort/pack/upload (or whatever the dispatcher does).
    // Stream parameter is templated to avoid requiring cuda_runtime.h here.
    // Return type is deduced from the dispatcher (TierResult, WarpTierResult, etc.).
    template <typename S>
    auto finish(S stream) -> decltype(dispatcher(chain.get_buffer(), stream)) {
        return dispatcher(chain.get_buffer(), stream);
    }
};

// Specialization: no dispatcher (collect-only, no finish)
template <typename Chain>
struct BuiltPipeline<Chain, void> {
    using Pkt = typename Chain::InputPacket;
    using DescType = DescOf<Pkt>;

    Chain chain;

    void push(DescType desc, int64_t offset_A, int64_t offset_B) {
        Pkt p;
        p.desc = std::move(desc);
        p.offset_A = offset_A;
        p.offset_B = offset_B;
        chain.push(std::move(p));
    }

    std::vector<Pkt>& collect() { return chain.get_buffer(); }
    const std::vector<Pkt>& collect() const { return chain.get_buffer(); }
};

// ============================================================================
// Builder: assembles stages into a nested chain at compile time
// ============================================================================

template <typename Desc, typename... Stages>
struct PipelineBuilder;

// Base builder: no stages yet
template <typename Desc>
struct PipelineBuilder<Desc> {
    using Pkt = Packet<Desc>;

    template <typename XForm>
    PipelineBuilder<Desc, XForm> transform(XForm xform) {
        return {std::move(xform)};
    }

    // Build without dispatcher (collect-only)
    BuiltPipeline<Collector<Pkt>> build() {
        return {Collector<Pkt>{}};
    }

    // Build with dispatcher (collect + finish)
    template <typename Disp>
    BuiltPipeline<Collector<Pkt>, Disp> build(Disp disp) {
        return {Collector<Pkt>{}, std::move(disp)};
    }
};

// Builder with one transformer stage
template <typename Desc, typename XForm>
struct PipelineBuilder<Desc, XForm> {
    using Pkt = Packet<Desc>;

    XForm xform;

    BuiltPipeline<TransformerNode<XForm, Collector<Pkt>>> build() {
        return {TransformerNode<XForm, Collector<Pkt>>{std::move(xform), Collector<Pkt>{}}};
    }

    template <typename Disp>
    BuiltPipeline<TransformerNode<XForm, Collector<Pkt>>, Disp> build(Disp disp) {
        return {TransformerNode<XForm, Collector<Pkt>>{std::move(xform), Collector<Pkt>{}},
                std::move(disp)};
    }
};

template <typename Desc>
PipelineBuilder<Desc> make_pipeline() {
    return {};
}

} // namespace bsptc
