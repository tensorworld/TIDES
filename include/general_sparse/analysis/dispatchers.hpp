#pragma once

#include "packet.hpp"
#include "../gpu_sparse_device.cuh"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

namespace bsptc {

// Result of dispatching a tier's packets: GPU-uploaded descs + offsets.
template <typename Desc>
struct TierResult {
    Desc* d_descs = nullptr;
    int num_descs = 0;
    int2* d_batch_ranges = nullptr;
    int num_batches = 0;
    bool use_batched = false;
    std::vector<int64_t> offset_A;
    std::vector<int64_t> offset_B;
};

// Sort packets in-place. Offsets travel with descs automatically.
template <typename Pkt, typename Less>
void sort_packets(std::vector<Pkt>& packets, Less less) {
    const int n = static_cast<int>(packets.size());
    if (n <= 1) return;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return less(packets[a].desc, packets[b].desc);
    });

    std::vector<Pkt> sorted(n);
    for (int i = 0; i < n; ++i)
        sorted[i] = std::move(packets[idx[i]]);
    packets = std::move(sorted);
}

// Greedy batch packing on sorted packets.
template <typename Pkt, typename WorkFn>
std::vector<int2> pack_batches(const std::vector<Pkt>& packets,
    WorkFn work_fn, int target_work, int max_per_batch)
{
    const int n = static_cast<int>(packets.size());
    if (n == 0) return {};

    std::vector<int2> ranges;
    int i = 0;
    while (i < n) {
        int start = i;
        int total = 0;
        int count = 0;
        while (i < n && count < max_per_batch && total < target_work) {
            total += work_fn(packets[i].desc);
            ++count;
            ++i;
        }
        ranges.push_back(make_int2(start, count));
    }
    return ranges;
}

// Extract descs + offsets from packets, upload descs to GPU.
template <typename Pkt>
TierResult<DescOf<Pkt>> upload_packets(
    const std::vector<Pkt>& packets,
    const std::vector<int2>& ranges,
    cudaStream_t stream)
{
    using Desc = DescOf<Pkt>;
    TierResult<Desc> result;
    const int n = static_cast<int>(packets.size());
    result.num_descs = n;
    if (n == 0) return result;

    // Extract descs and offsets
    std::vector<Desc> h_descs(n);
    result.offset_A.resize(n);
    result.offset_B.resize(n);
    for (int i = 0; i < n; ++i) {
        h_descs[i] = packets[i].desc;
        result.offset_A[i] = packets[i].offset_A;
        result.offset_B[i] = packets[i].offset_B;
    }

    // Upload descs
    check_cuda(cudaMalloc(&result.d_descs, n * sizeof(Desc)), "malloc descs");
    check_cuda(cudaMemcpyAsync(result.d_descs, h_descs.data(),
                               n * sizeof(Desc),
                               cudaMemcpyHostToDevice, stream),
               "memcpy descs");

    // Upload batch ranges if provided
    if (!ranges.empty()) {
        result.use_batched = true;
        result.num_batches = static_cast<int>(ranges.size());
        check_cuda(cudaMalloc(&result.d_batch_ranges,
                              ranges.size() * sizeof(int2)),
                   "malloc batch_ranges");
        check_cuda(cudaMemcpyAsync(result.d_batch_ranges, ranges.data(),
                                   ranges.size() * sizeof(int2),
                                   cudaMemcpyHostToDevice, stream),
                   "memcpy batch_ranges");
    }

    return result;
}

// ============================================================================
// Dispatcher policies (passed to BuiltPipeline via build(dispatcher))
// ============================================================================

// Upload-only: no sort, no batch packing. Just extract and upload.
struct UploadDispatcher {
    template <typename Pkt>
    TierResult<DescOf<Pkt>> operator()(std::vector<Pkt>& packets, cudaStream_t stream) const {
        return upload_packets(packets, {}, stream);
    }
};

// Sort → pack → upload. If fewer than min_for_batching packets, skips sort/pack.
template <typename SortFn, typename WorkFn>
struct SortPackDispatcher {
    SortFn sort_fn;
    WorkFn work_fn;
    int target_work;
    int max_per_batch;
    int min_for_batching;

    template <typename Pkt>
    TierResult<DescOf<Pkt>> operator()(std::vector<Pkt>& packets, cudaStream_t stream) const {
        std::vector<int2> ranges;
        if (static_cast<int>(packets.size()) >= min_for_batching) {
            sort_packets(packets, sort_fn);
            ranges = pack_batches(packets, work_fn, target_work, max_per_batch);
        }
        return upload_packets(packets, ranges, stream);
    }
};

template <typename SortFn, typename WorkFn>
SortPackDispatcher<SortFn, WorkFn> make_sort_pack_dispatcher(
    SortFn sort_fn, WorkFn work_fn,
    int target_work, int max_per_batch, int min_for_batching)
{
    return {std::move(sort_fn), std::move(work_fn),
            target_work, max_per_batch, min_for_batching};
}


} // namespace bsptc
