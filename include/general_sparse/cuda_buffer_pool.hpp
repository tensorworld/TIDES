#pragma once

// Reusable pool of CUDA buffers (pinned host or device), keyed by capacity.
// cudaMallocHost costs ~1 ms per call (OS page pinning) and dominated
// pipeline_allocate in the setup baseline (docs/setup_optimization.md).
// Buffers acquired here are returned via release() and reused by later
// pipeline invocations instead of being freed.
//
// Not thread-safe — the host pipeline is single-threaded.
// Device-aware: device buffers are stamped with their allocating device and
// only handed out again on that device (a device-0 pointer must not reach
// kernels on device 1). Pinned host buffers are portable across devices.
// Idle pool memory is bounded by trim(); remaining buffers are intentionally
// not freed at process exit (static-destructor cudaFree would race CUDA
// context teardown; the driver reclaims everything).

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

#include "gpu_sparse_device.cuh"

namespace bsptc {

class CudaBufferPool {
public:
    enum class Kind { Pinned, Device };

    explicit CudaBufferPool(Kind kind) : kind_(kind) {}

    void* acquire(size_t bytes) {
        if (bytes == 0) return nullptr;
        int dev = 0;
        if (kind_ == Kind::Device)
            check_cuda(cudaGetDevice(&dev), "buffer_pool: cudaGetDevice");
        // Smallest free entry that fits (device buffers: current device only)
        int best = -1;
        for (int i = 0; i < static_cast<int>(free_.size()); ++i) {
            if (free_[i].cap >= bytes && free_[i].dev == dev &&
                (best < 0 || free_[i].cap < free_[best].cap))
                best = i;
        }
        Entry e;
        if (best >= 0) {
            e = free_[best];
            free_[best] = free_.back();
            free_.pop_back();
        } else {
            e.cap = round_up_pow2(bytes);
            e.dev = dev;
            if (kind_ == Kind::Pinned)
                check_cuda(cudaMallocHost(&e.ptr, e.cap), "buffer_pool: cudaMallocHost");
            else
                check_cuda(cudaMalloc(&e.ptr, e.cap), "buffer_pool: cudaMalloc");
        }
        live_.push_back(e);
        return e.ptr;
    }

    // Return a buffer to the pool for reuse. The caller must guarantee no
    // device work still reads/writes the buffer (unlike cudaFreeHost, this
    // does NOT synchronize).
    void release(void* p) {
        if (!p) return;
        for (int i = 0; i < static_cast<int>(live_.size()); ++i) {
            if (live_[i].ptr == p) {
                free_.push_back(live_[i]);
                live_[i] = live_.back();
                live_.pop_back();
                trim();
                return;
            }
        }
        // Not pool-owned: free directly (defensive; shouldn't happen).
        if (kind_ == Kind::Pinned) cudaFreeHost(p);
        else cudaFree(p);
    }

private:
    // dev: allocating device for Kind::Device (0 for pinned — portable).
    // cudaFree on another device's pointer is legal under UVA, so trim()
    // needs no device switch.
    struct Entry { void* ptr = nullptr; size_t cap = 0; int dev = 0; };

    static size_t round_up_pow2(size_t n) {
        size_t c = 4096;
        while (c < n) c <<= 1;
        return c;
    }

    // Bound idle pool memory: drop largest free entries beyond the cap.
    void trim() {
        constexpr size_t MAX_FREE_BYTES = 64ull << 20;  // per pool
        size_t total = 0;
        for (const auto& e : free_) total += e.cap;
        while (total > MAX_FREE_BYTES && !free_.empty()) {
            int largest = 0;
            for (int i = 1; i < static_cast<int>(free_.size()); ++i)
                if (free_[i].cap > free_[largest].cap) largest = i;
            total -= free_[largest].cap;
            if (kind_ == Kind::Pinned) cudaFreeHost(free_[largest].ptr);
            else cudaFree(free_[largest].ptr);
            free_[largest] = free_.back();
            free_.pop_back();
        }
    }

    Kind kind_;
    std::vector<Entry> live_;
    std::vector<Entry> free_;
};

// Enable caching in the device's default stream-ordered memory pool so
// cudaMallocAsync reuses freed GB-scale buffers across pipeline invocations
// (~2 us warm vs ~2 ms/GB for a fresh cudaMalloc). The driver reclaims idle
// pool memory whenever a plain cudaMalloc would otherwise fail (verified
// empirically), so the cache cannot starve other allocators (e.g. cuTENSOR).
// Configured once per device (not thread-safe, like the pools above).
inline void ensure_mempool_caching() {
    constexpr int MAX_DEVICES = 64;
    static bool configured[MAX_DEVICES] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= MAX_DEVICES ||
        configured[dev])
        return;
    configured[dev] = true;
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
        uint64_t thresh = ~0ull;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thresh);
    }
}

// Stream-ordered allocation on the legacy default stream. Memory is freed
// with plain cudaFree (valid for cudaMallocAsync allocations), so existing
// destructor paths need no change.
inline cudaError_t malloc_device_cached(void** ptr, size_t bytes) {
    ensure_mempool_caching();
    return cudaMallocAsync(ptr, bytes, static_cast<cudaStream_t>(nullptr));
}

// Process-wide pools for CUTLASS grouped-GEMM arg buffers.
// Heap-allocated singletons: never destructed, so no exit-order issues.
inline CudaBufferPool& pinned_args_pool() {
    static CudaBufferPool* pool = new CudaBufferPool(CudaBufferPool::Kind::Pinned);
    return *pool;
}

inline CudaBufferPool& device_args_pool() {
    static CudaBufferPool* pool = new CudaBufferPool(CudaBufferPool::Kind::Device);
    return *pool;
}

} // namespace bsptc
