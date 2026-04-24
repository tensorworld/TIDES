#pragma once

#include "../block_sparse_tensor.hpp"
#include "../block_utils.hpp"
#include "../einsum.hpp"
#include "../layout_utils.hpp"
#include "../cutensor_utils.cuh"

#include <cuda_runtime.h>
#include <cutensor.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// cuTENSOR block-sparse contraction wrapper.
// Builds block-sparse descriptors from our BlockSparseTensor format,
// executes cutensorBlockSparseContract, and returns block-sparse output.
//
// cuTENSOR's block-sparse beta API (2.4.1–2.5.0) does not support batch
// (Hadamard) modes — modes that appear in all three operands (A, B, C/D).
// We work around this by looping over batch slices: for each unique batch
// coordinate, extract sub-tensors without the batch dimensions and run a
// separate block-sparse contraction per slice.
namespace bsptc {

template <typename T>
struct CutensorBlockSparseResult {
    float kernel_ms;
    float setup_ms = 0;         // descriptor + plan creation (excludes malloc/memcpy)
    int64_t peak_gpu_bytes = 0;  // estimated peak GPU memory (inputs + output + workspace)

    // Block-sparse output (always populated)
    std::vector<std::vector<int64_t>> block_coords;  // per-block section coords (output label order)
    std::vector<T> block_data;                        // flat packed block data (col-major per block)
    std::vector<int64_t> block_offsets;               // per-block offset into block_data
    std::vector<int64_t> block_volumes;               // per-block element count
    int64_t total_block_elements = 0;
};

namespace detail_bs {

inline void check_cutensor(cutensorStatus_t status, const char* msg) {
    if (status != CUTENSOR_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " +
                                 cutensorGetErrorString(status));
    }
}

// RAII wrapper for a C handle: calls Destroy(h) on scope exit.
template <typename H, auto Destroy>
struct CHandle {
    H h{};
    CHandle() = default;
    ~CHandle() { if (h) Destroy(h); }
    CHandle(const CHandle&) = delete;
    CHandle& operator=(const CHandle&) = delete;
};

template <typename T>
struct BSDescriptor {
    cutensorBlockSparseTensorDescriptor_t desc{};
    std::vector<const void*> d_block_ptrs;
    void* d_buffer = nullptr;
    bool owns_buffer = true;    // false when using external device memory
    int64_t buffer_bytes = 0;   // size of d_buffer allocation
    uint64_t num_blocks = 0;

    BSDescriptor() = default;
    ~BSDescriptor() {
        if (desc) cutensorDestroyBlockSparseTensorDescriptor(desc);
        if (owns_buffer && d_buffer) cudaFree(d_buffer);
    }
    BSDescriptor(BSDescriptor&& o) noexcept
        : desc(o.desc), d_block_ptrs(std::move(o.d_block_ptrs)),
          d_buffer(o.d_buffer), owns_buffer(o.owns_buffer),
          buffer_bytes(o.buffer_bytes), num_blocks(o.num_blocks)
    { o.desc = {}; o.d_buffer = nullptr; }
    BSDescriptor& operator=(BSDescriptor&& o) noexcept {
        if (this != &o) {
            if (desc) cutensorDestroyBlockSparseTensorDescriptor(desc);
            if (owns_buffer && d_buffer) cudaFree(d_buffer);
            desc = o.desc; o.desc = {};
            d_block_ptrs = std::move(o.d_block_ptrs);
            d_buffer = o.d_buffer; o.d_buffer = nullptr;
            owns_buffer = o.owns_buffer;
            num_blocks = o.num_blocks;
        }
        return *this;
    }
    BSDescriptor(const BSDescriptor&) = delete;
    BSDescriptor& operator=(const BSDescriptor&) = delete;
};

template <typename T>
void repack_row_to_col(const T* src, T* dst,
                       const std::vector<int64_t>& shape) {
    const int ndim = static_cast<int>(shape.size());
    int64_t volume = 1;
    for (int64_t s : shape) volume *= s;

    std::vector<int64_t> rm_strides(ndim);
    std::vector<int64_t> cm_strides(ndim);
    if (ndim > 0) {
        rm_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d)
            rm_strides[d] = rm_strides[d + 1] * shape[d + 1];
        cm_strides[0] = 1;
        for (int d = 1; d < ndim; ++d)
            cm_strides[d] = cm_strides[d - 1] * shape[d - 1];
    }

    for (int64_t i = 0; i < volume; ++i) {
        int64_t remaining = i;
        int64_t cm_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = remaining / rm_strides[d];
            remaining -= coord * rm_strides[d];
            cm_offset += coord * cm_strides[d];
        }
        dst[cm_offset] = src[i];
    }
}

inline void build_extent_arrays(
    const BlockSparseTensor& bst,
    std::vector<int64_t>& extent_out,
    std::vector<uint32_t>& sections_per_mode_out)
{
    extent_out.clear();
    sections_per_mode_out.clear();
    for (char lbl : bst.meta.indices) {
        const auto& sizes = bst.block_sizes.at(lbl);
        sections_per_mode_out.push_back(static_cast<uint32_t>(sizes.size()));
        for (int64_t s : sizes) {
            extent_out.push_back(s);
        }
    }
}

inline std::vector<int32_t> flatten_block_coords(
    const BlockSparseTensor& bst)
{
    const int ndim = bst.meta.ndim;
    std::vector<int32_t> flat;
    flat.reserve(bst.block_coords.size() * ndim);
    for (const auto& coord : bst.block_coords) {
        for (int64_t c : coord) {
            flat.push_back(static_cast<int32_t>(c));
        }
    }
    return flat;
}

// Create a cuTENSOR block-sparse descriptor.
// If d_external_values is non-null, uses that device buffer (caller-owned)
// instead of uploading from host.
template <typename T>
BSDescriptor<T> create_bs_descriptor(
    cutensorHandle_t handle,
    const BlockSparseTensor& bst,
    const T* d_external_values = nullptr)
{
    BSDescriptor<T> result;
    const uint32_t numModes = static_cast<uint32_t>(bst.meta.ndim);
    const int ndim = bst.meta.ndim;
    result.num_blocks = static_cast<uint64_t>(bst.block_coords.size());

    std::vector<int64_t> extents;
    std::vector<uint32_t> sections_per_mode;
    build_extent_arrays(bst, extents, sections_per_mode);
    auto flat_coords = flatten_block_coords(bst);

    // Compute per-block row-major strides so we can skip repack_row_to_col.
    // Layout: flat array of numModes * numBlocks int64_t values,
    // block 0 strides first, then block 1, etc.
    const auto block_offsets = block_utils::compute_block_offsets(bst);
    std::vector<int64_t> all_strides;
    all_strides.reserve(static_cast<size_t>(ndim) * result.num_blocks);
    int64_t total_elements = 0;

    for (const auto& coord : bst.block_coords) {
        auto shape = block_utils::block_shape_for_coord(bst, coord);
        // Row-major strides: stride[last] = 1, stride[i] = prod(shape[i+1..])
        std::vector<int64_t> rm_strides(ndim);
        if (ndim > 0) {
            rm_strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; --d)
                rm_strides[d] = rm_strides[d + 1] * shape[d + 1];
        }
        all_strides.insert(all_strides.end(), rm_strides.begin(), rm_strides.end());

        int64_t vol = 1;
        for (int64_t s : shape) vol *= s;
        total_elements += vol;
    }

    check_cutensor(
        cutensorCreateBlockSparseTensorDescriptor(
            handle, &result.desc,
            numModes,
            result.num_blocks,
            sections_per_mode.data(),
            extents.data(),
            flat_coords.data(),
            all_strides.data(),
            to_cutensor_type(bst.meta.dtype)),
        "cutensorCreateBlockSparseTensorDescriptor");

    result.buffer_bytes = total_elements * static_cast<int64_t>(sizeof(T));

    if (d_external_values) {
        // Borrow caller's device buffer — no upload, no ownership.
        result.d_buffer = const_cast<void*>(static_cast<const void*>(d_external_values));
        result.owns_buffer = false;
    } else {
        // Upload from host.
        result.d_buffer = nullptr;
        if (total_elements > 0) {
            cudaMalloc(&result.d_buffer, total_elements * sizeof(T));
            const T* src = bst.host_data.data_as<T>();
            cudaMemcpy(result.d_buffer, src, total_elements * sizeof(T),
                       cudaMemcpyHostToDevice);
        }
    }

    // Set up per-block device pointers into the buffer.
    result.d_block_ptrs.resize(result.num_blocks);
    for (size_t blk_idx = 0; blk_idx < result.num_blocks; ++blk_idx) {
        T* d_ptr = static_cast<T*>(result.d_buffer) + block_offsets[blk_idx];
        result.d_block_ptrs[blk_idx] = static_cast<const void*>(d_ptr);
    }

    return result;
}

// Extract an element-level batch slice from a BlockSparseTensor.
// For each block whose batch section coordinates match `batch_section_coord`,
// extract the sub-array at `batch_local_indices` (one per batch dim) along the
// batch dimensions. Returns a BST with the batch dimensions removed and only
// the sliced data.
template <typename T>
BlockSparseTensor slice_bst_for_batch(
    const BlockSparseTensor& bst,
    const std::vector<int>& batch_dims,                // positions of batch dims in bst
    const std::vector<int64_t>& batch_section_coord,   // block section indices for batch dims
    const std::vector<int64_t>& batch_local_indices)   // element-within-section indices
{
    const int orig_rank = bst.meta.ndim;

    // Build mask of which dims to keep
    std::vector<bool> is_batch(orig_rank, false);
    for (int d : batch_dims) is_batch[d] = true;

    // New metadata (without batch dims)
    TensorMeta meta;
    meta.name = bst.meta.name;
    meta.dtype = bst.meta.dtype;
    std::vector<int> new_num_blocks;
    std::map<char, std::vector<int64_t>> new_block_sizes;

    for (int d = 0; d < orig_rank; ++d) {
        if (is_batch[d]) continue;
        char lbl = bst.meta.indices[d];
        meta.indices.push_back(lbl);
        new_num_blocks.push_back(bst.num_blocks[d]);
        new_block_sizes[lbl] = bst.block_sizes.at(lbl);
    }
    meta.ndim = static_cast<int16_t>(meta.indices.size());

    // Filter blocks by batch section coordinate
    std::set<std::vector<int64_t>> new_coords;
    struct KeptBlock {
        size_t orig_idx;
        std::vector<int64_t> orig_shape;
        std::vector<int64_t> new_coord;
    };
    std::vector<KeptBlock> kept_blocks;

    const auto block_offsets = block_utils::compute_block_offsets(bst);
    size_t blk_idx = 0;
    for (const auto& coord : bst.block_coords) {
        bool match = true;
        for (size_t bi = 0; bi < batch_dims.size(); ++bi) {
            if (coord[batch_dims[bi]] != batch_section_coord[bi]) {
                match = false;
                break;
            }
        }
        if (match) {
            std::vector<int64_t> new_coord;
            for (int d = 0; d < orig_rank; ++d) {
                if (!is_batch[d]) new_coord.push_back(coord[d]);
            }
            new_coords.insert(new_coord);
            auto shape = block_utils::block_shape_for_coord(bst, coord);
            kept_blocks.push_back({blk_idx, std::move(shape), std::move(new_coord)});
        }
        ++blk_idx;
    }

    BlockSparseTensor result(meta, new_num_blocks, new_block_sizes,
                             new_coords, bst.layout);

    // Compute total sliced elements
    size_t total_sliced = 0;
    for (const auto& kb : kept_blocks) {
        int64_t vol = 1;
        for (int d = 0; d < orig_rank; ++d) {
            if (!is_batch[d]) vol *= kb.orig_shape[d];
        }
        total_sliced += static_cast<size_t>(vol);
    }

    result.host_data.set_data_type(meta.dtype);
    result.host_data.allocate(total_sliced);
    T* dst = result.host_data.data_as<T>();
    const T* src = bst.host_data.data_as<T>();

    // For each kept block, extract the element-level slice
    size_t dst_offset = 0;
    for (const auto& kb : kept_blocks) {
        const auto& shape = kb.orig_shape;
        int64_t src_base = static_cast<int64_t>(block_offsets[kb.orig_idx]);

        // Compute row-major strides for original block
        std::vector<int64_t> rm_strides(orig_rank);
        rm_strides[orig_rank - 1] = 1;
        for (int d = orig_rank - 2; d >= 0; --d)
            rm_strides[d] = rm_strides[d + 1] * shape[d + 1];

        // Compute the fixed offset contribution from batch dimensions
        int64_t batch_fixed_offset = 0;
        for (size_t bi = 0; bi < batch_dims.size(); ++bi) {
            batch_fixed_offset += batch_local_indices[bi] * rm_strides[batch_dims[bi]];
        }

        // New shape (without batch dims)
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_rm_strides;
        for (int d = 0; d < orig_rank; ++d) {
            if (!is_batch[d]) {
                new_shape.push_back(shape[d]);
                new_rm_strides.push_back(rm_strides[d]);
            }
        }

        int new_rank = static_cast<int>(new_shape.size());
        int64_t slice_vol = 1;
        for (int64_t s : new_shape) slice_vol *= s;

        // Compute row-major strides for the new (destination) block
        std::vector<int64_t> dst_rm_strides(new_rank);
        if (new_rank > 0) {
            dst_rm_strides[new_rank - 1] = 1;
            for (int d = new_rank - 2; d >= 0; --d)
                dst_rm_strides[d] = dst_rm_strides[d + 1] * new_shape[d + 1];
        }

        // Copy elements: iterate in row-major order of the new block
        for (int64_t i = 0; i < slice_vol; ++i) {
            // Decompose i into multi-index using new row-major strides
            int64_t remaining = i;
            int64_t src_off = batch_fixed_offset;
            for (int d = 0; d < new_rank; ++d) {
                int64_t idx = remaining / dst_rm_strides[d];
                remaining -= idx * dst_rm_strides[d];
                src_off += idx * new_rm_strides[d];
            }
            dst[dst_offset + i] = src[src_base + src_off];
        }

        dst_offset += static_cast<size_t>(slice_vol);
    }

    return result;
}

} // namespace detail_bs

inline int label_pos(const std::vector<char>& labels, char c) {
    for (size_t i = 0; i < labels.size(); ++i)
        if (labels[i] == c) return static_cast<int>(i);
    return -1;
}

// Compute the output block sparsity pattern from A and B block coordinates.
// For each pair of A/B blocks sharing all shared-index coordinates (contracted + batch),
// record the output block coordinate in output label order.
// Returns sorted, deduplicated output block coordinates.
inline std::vector<std::vector<int64_t>> compute_output_block_coords(
    const BlockSparseTensor& A_bst,
    const BlockSparseTensor& B_bst,
    const EinsumExpression& expr)
{
    EinsumIndexSets index_sets;
    create_einsum_index_sets(expr, index_sets);

    // Match on ALL shared indices (contracted + batch)
    std::vector<int> shared_pos_A, shared_pos_B;
    for (char c : index_sets.contracted_indices) {
        shared_pos_A.push_back(label_pos(A_bst.meta.indices, c));
        shared_pos_B.push_back(label_pos(B_bst.meta.indices, c));
    }
    for (char c : index_sets.batch_indices) {
        shared_pos_A.push_back(label_pos(A_bst.meta.indices, c));
        shared_pos_B.push_back(label_pos(B_bst.meta.indices, c));
    }

    struct OutputDimSource {
        int tensor;  // 0=A, 1=B
        int dim;
    };
    std::vector<OutputDimSource> output_sources;
    for (char c : expr.output_labels) {
        int pa = label_pos(A_bst.meta.indices, c);
        if (pa >= 0) {
            output_sources.push_back({0, pa});
        } else {
            int pb = label_pos(B_bst.meta.indices, c);
            output_sources.push_back({1, pb});
        }
    }

    std::map<std::vector<int64_t>, std::vector<const std::vector<int64_t>*>> a_by_shared;
    for (const auto& coord_a : A_bst.block_coords) {
        std::vector<int64_t> key;
        for (int p : shared_pos_A) key.push_back(coord_a[p]);
        a_by_shared[key].push_back(&coord_a);
    }

    std::set<std::vector<int64_t>> output_coords_set;
    for (const auto& coord_b : B_bst.block_coords) {
        std::vector<int64_t> key;
        for (int p : shared_pos_B) key.push_back(coord_b[p]);

        auto it = a_by_shared.find(key);
        if (it == a_by_shared.end()) continue;

        for (const auto* coord_a_ptr : it->second) {
            std::vector<int64_t> out_coord;
            out_coord.reserve(output_sources.size());
            for (const auto& src : output_sources) {
                if (src.tensor == 0)
                    out_coord.push_back((*coord_a_ptr)[src.dim]);
                else
                    out_coord.push_back(coord_b[src.dim]);
            }
            output_coords_set.insert(std::move(out_coord));
        }
    }

    return std::vector<std::vector<int64_t>>(
        output_coords_set.begin(), output_coords_set.end());
}

// Build output BlockSparseTensor metadata (no data) for the output descriptor.
inline BlockSparseTensor build_output_bst_meta(
    const BlockSparseTensor& A_bst,
    const BlockSparseTensor& B_bst,
    const EinsumExpression& expr,
    const std::vector<std::vector<int64_t>>& output_coords)
{
    std::map<char, std::vector<int64_t>> out_block_sizes;
    std::vector<int> out_num_blocks;

    for (char c : expr.output_labels) {
        auto it = A_bst.block_sizes.find(c);
        if (it != A_bst.block_sizes.end()) {
            out_block_sizes[c] = it->second;
            for (size_t i = 0; i < A_bst.meta.indices.size(); ++i) {
                if (A_bst.meta.indices[i] == c) {
                    out_num_blocks.push_back(A_bst.num_blocks[i]);
                    break;
                }
            }
        } else {
            auto it2 = B_bst.block_sizes.find(c);
            out_block_sizes[c] = it2->second;
            for (size_t i = 0; i < B_bst.meta.indices.size(); ++i) {
                if (B_bst.meta.indices[i] == c) {
                    out_num_blocks.push_back(B_bst.num_blocks[i]);
                    break;
                }
            }
        }
    }

    std::set<std::vector<int64_t>> coords_set(
        output_coords.begin(), output_coords.end());

    TensorMeta meta;
    meta.name = "output";
    meta.ndim = static_cast<int16_t>(expr.output_labels.size());
    meta.indices = std::vector<char>(expr.output_labels.begin(),
                                     expr.output_labels.end());
    meta.dtype = A_bst.meta.dtype;

    BlockSparseTensor out_bst(meta, out_num_blocks, out_block_sizes,
                              coords_set, LayoutType::ColumnMajor);
    return out_bst;
}

// Compute per-block volumes and offsets for a list of output block coords.
// Returns {volumes, total_elements}.
inline std::pair<std::vector<int64_t>, int64_t> compute_block_volumes_and_offsets(
    const BlockSparseTensor& out_bst,
    const std::vector<std::vector<int64_t>>& output_coords,
    std::vector<int64_t>& offsets_out)
{
    std::vector<int64_t> volumes;
    volumes.reserve(output_coords.size());
    offsets_out.resize(output_coords.size());
    int64_t total = 0;
    for (size_t i = 0; i < output_coords.size(); ++i) {
        auto shape = block_utils::block_shape_for_coord(out_bst, output_coords[i]);
        int64_t vol = 1;
        for (int64_t s : shape) vol *= s;
        offsets_out[i] = total;
        volumes.push_back(vol);
        total += vol;
    }
    return {std::move(volumes), total};
}

// Populate CutensorBlockSparseResult fields from precomputed volumes/offsets.
// If d_out_buffer is non-null, copies GPU data to result.block_data.
template <typename T>
void populate_result_fields(
    CutensorBlockSparseResult<T>& result,
    const std::vector<std::vector<int64_t>>& coords,
    const std::vector<int64_t>& volumes,
    const std::vector<int64_t>& offsets,
    int64_t total_elements,
    const void* d_out_buffer)
{
    result.block_coords = coords;
    result.block_volumes = volumes;
    result.block_offsets = offsets;
    result.total_block_elements = total_elements;
    result.block_data.resize(static_cast<size_t>(total_elements));
    if (d_out_buffer && total_elements > 0) {
        cudaMemcpy(result.block_data.data(), d_out_buffer,
                   total_elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
}

// Run a single (non-batched) block-sparse contraction.
// A_bst and B_bst must not have any batch modes shared with the output.
// If d_A_values / d_B_values are non-null, use those device buffers instead of uploading.
template <typename T>
CutensorBlockSparseResult<T> run_cutensor_block_sparse_single(
    const BlockSparseTensor& A_bst,
    const BlockSparseTensor& B_bst,
    const EinsumExpression& expr,
    const T* d_A_values = nullptr,
    const T* d_B_values = nullptr,
    const std::vector<std::vector<int64_t>>* precomputed_output_coords = nullptr)
{
    using detail_bs::CHandle;
    CutensorBlockSparseResult<T> result;
    result.kernel_ms = 0.0f;
    using setup_clock = std::chrono::high_resolution_clock;

    cutensorHandle_t handle = get_cached_cutensor_handle();

    auto a_desc = detail_bs::create_bs_descriptor<T>(handle, A_bst, d_A_values);
    auto b_desc = detail_bs::create_bs_descriptor<T>(handle, B_bst, d_B_values);

    auto output_coords = precomputed_output_coords
        ? *precomputed_output_coords
        : compute_output_block_coords(A_bst, B_bst, expr);
    if (output_coords.empty()) return result;

    auto out_bst = build_output_bst_meta(A_bst, B_bst, expr, output_coords);
    const uint64_t out_nnz = static_cast<uint64_t>(output_coords.size());

    std::vector<int64_t> out_extents;
    std::vector<uint32_t> out_sections;
    detail_bs::build_extent_arrays(out_bst, out_extents, out_sections);
    auto out_flat_coords = detail_bs::flatten_block_coords(out_bst);

    CHandle<cutensorBlockSparseTensorDescriptor_t,
            cutensorDestroyBlockSparseTensorDescriptor> out_desc;
    detail_bs::check_cutensor(
        cutensorCreateBlockSparseTensorDescriptor(
            handle, &out_desc.h,
            static_cast<uint32_t>(out_bst.meta.ndim), out_nnz,
            out_sections.data(), out_extents.data(),
            out_flat_coords.data(), nullptr,
            to_cutensor_type(out_bst.meta.dtype)),
        "cutensorCreateBlockSparseTensorDescriptor (output)");

    std::vector<int64_t> out_offsets;
    auto [out_block_volumes, out_total_elements] =
        compute_block_volumes_and_offsets(out_bst, output_coords, out_offsets);

    CHandle<void*, cudaFree> d_out_buffer;
    if (out_total_elements > 0) {
        cudaMalloc(&d_out_buffer.h, out_total_elements * sizeof(T));
        cudaMemset(d_out_buffer.h, 0, out_total_elements * sizeof(T));
    }

    std::vector<const void*> d_C_ptrs(out_nnz);
    std::vector<void*> d_D_ptrs(out_nnz);
    int64_t ptr_offset = 0;
    for (uint64_t i = 0; i < out_nnz; ++i) {
        T* ptr = static_cast<T*>(d_out_buffer.h) + ptr_offset;
        d_C_ptrs[i] = static_cast<const void*>(ptr);
        d_D_ptrs[i] = static_cast<void*>(ptr);
        ptr_offset += out_block_volumes[i];
    }

    std::vector<int32_t> mA, mB, mO;
    for (char c : A_bst.meta.indices) mA.push_back(static_cast<int32_t>(c));
    for (char c : B_bst.meta.indices) mB.push_back(static_cast<int32_t>(c));
    for (char c : expr.output_labels) mO.push_back(static_cast<int32_t>(c));

    auto compute_desc = to_cutensor_compute(A_bst.meta.dtype);

    // --- TIMED: descriptor + plan creation (no memory ops) ---
    auto t_setup_start = setup_clock::now();

    CHandle<cutensorOperationDescriptor_t,
            cutensorDestroyOperationDescriptor> opDesc;
    detail_bs::check_cutensor(
        cutensorCreateBlockSparseContraction(
            handle, &opDesc.h,
            a_desc.desc, mA.data(), CUTENSOR_OP_IDENTITY,
            b_desc.desc, mB.data(), CUTENSOR_OP_IDENTITY,
            out_desc.h, mO.data(), CUTENSOR_OP_IDENTITY,
            out_desc.h, mO.data(),
            compute_desc),
        "cutensorCreateBlockSparseContraction");

    CHandle<cutensorPlanPreference_t, cutensorDestroyPlanPreference> pref;
    cutensorCreatePlanPreference(handle, &pref.h,
                                 CUTENSOR_ALGO_DEFAULT,
                                 CUTENSOR_JIT_MODE_NONE);

    uint64_t ws_size = 0;
    cutensorEstimateWorkspaceSize(handle, opDesc.h, pref.h,
                                  CUTENSOR_WORKSPACE_DEFAULT, &ws_size);

    // Check whether output + workspace will fit in free GPU memory.
    // Query actual free memory rather than using a conservative heuristic.
    {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        uint64_t need = static_cast<uint64_t>(out_total_elements) * sizeof(T) + ws_size;
        if (need > free_bytes) {
            auto mb = [](uint64_t b) { return static_cast<double>(b) / (1024.0*1024.0); };
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "cuTENSOR OOM: need %.0f MB (output=%.0f + workspace=%.0f), "
                     "free=%.0f MB",
                     mb(need),
                     mb(static_cast<uint64_t>(out_total_elements) * sizeof(T)),
                     mb(ws_size), mb(free_bytes));
            throw std::runtime_error(buf);
        }
    }

    CHandle<cutensorPlan_t, cutensorDestroyPlan> plan;
    detail_bs::check_cutensor(
        cutensorCreatePlan(handle, &plan.h, opDesc.h, pref.h, ws_size),
        "cutensorCreatePlan (block sparse)");

    auto t_setup_end = setup_clock::now();
    result.setup_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(t_setup_end - t_setup_start).count());

    CHandle<void*, cudaFree> d_workspace;
    if (ws_size > 0) cudaMalloc(&d_workspace.h, ws_size);

    T alpha_val = static_cast<T>(1);
    T beta_val = static_cast<T>(0);

    CHandle<cudaStream_t, cudaStreamDestroy> stream;
    cudaStreamCreate(&stream.h);

    CHandle<cudaEvent_t, cudaEventDestroy> ev0, ev1;
    cudaEventCreate(&ev0.h);
    cudaEventCreate(&ev1.h);

    cudaEventRecord(ev0.h, stream.h);
    detail_bs::check_cutensor(
        cutensorBlockSparseContract(
            handle, plan.h,
            &alpha_val, a_desc.d_block_ptrs.data(),
                        b_desc.d_block_ptrs.data(),
            &beta_val,  d_C_ptrs.data(),
                        d_D_ptrs.data(),
            d_workspace.h, ws_size, stream.h),
        "cutensorBlockSparseContract");
    cudaEventRecord(ev1.h, stream.h);
    cudaEventSynchronize(ev1.h);
    cudaEventElapsedTime(&result.kernel_ms, ev0.h, ev1.h);

    populate_result_fields(result, output_coords, out_block_volumes,
                           out_offsets, out_total_elements, d_out_buffer.h);

    // Peak GPU memory: inputs + output + workspace (all live simultaneously)
    result.peak_gpu_bytes = a_desc.buffer_bytes + b_desc.buffer_bytes
                          + out_total_elements * static_cast<int64_t>(sizeof(T))
                          + static_cast<int64_t>(ws_size);

    return result;
}

// Prepared cuTENSOR contract: setup once, run() many times for benchmarking.
// Holds all cuTENSOR objects (plan, descriptors, workspace, device buffers).
// run() calls cutensorBlockSparseContract + cudaStreamSynchronize (no setup).
template <typename T>
struct CutensorPreparedContract {
    cutensorHandle_t handle = nullptr;
    cutensorPlan_t plan = nullptr;
    cutensorBlockSparseTensorDescriptor_t out_desc = nullptr;
    detail_bs::BSDescriptor<T> a_desc, b_desc;
    std::vector<const void*> d_C_ptrs;
    std::vector<void*> d_D_ptrs;
    void* d_out_buffer = nullptr;
    void* d_workspace = nullptr;
    uint64_t ws_size = 0;
    cudaStream_t stream = nullptr;

    CutensorPreparedContract() = default;
    ~CutensorPreparedContract() {
        if (plan) cutensorDestroyPlan(plan);
        if (out_desc) cutensorDestroyBlockSparseTensorDescriptor(out_desc);
        if (d_out_buffer) cudaFree(d_out_buffer);
        if (d_workspace) cudaFree(d_workspace);
        if (stream) cudaStreamDestroy(stream);
    }
    CutensorPreparedContract(const CutensorPreparedContract&) = delete;
    CutensorPreparedContract& operator=(const CutensorPreparedContract&) = delete;
    CutensorPreparedContract(CutensorPreparedContract&& o) noexcept
        : handle(o.handle), plan(o.plan), out_desc(o.out_desc),
          a_desc(std::move(o.a_desc)), b_desc(std::move(o.b_desc)),
          d_C_ptrs(std::move(o.d_C_ptrs)), d_D_ptrs(std::move(o.d_D_ptrs)),
          d_out_buffer(o.d_out_buffer), d_workspace(o.d_workspace),
          ws_size(o.ws_size), stream(o.stream)
    { o.plan = nullptr; o.out_desc = nullptr; o.d_out_buffer = nullptr;
      o.d_workspace = nullptr; o.stream = nullptr; }
    CutensorPreparedContract& operator=(CutensorPreparedContract&& o) noexcept {
        if (this != &o) {
            if (plan) cutensorDestroyPlan(plan);
            if (out_desc) cutensorDestroyBlockSparseTensorDescriptor(out_desc);
            if (d_out_buffer) cudaFree(d_out_buffer);
            if (d_workspace) cudaFree(d_workspace);
            if (stream) cudaStreamDestroy(stream);
            handle = o.handle; plan = o.plan; out_desc = o.out_desc;
            a_desc = std::move(o.a_desc); b_desc = std::move(o.b_desc);
            d_C_ptrs = std::move(o.d_C_ptrs); d_D_ptrs = std::move(o.d_D_ptrs);
            d_out_buffer = o.d_out_buffer; d_workspace = o.d_workspace;
            ws_size = o.ws_size; stream = o.stream;
            o.plan = nullptr; o.out_desc = nullptr; o.d_out_buffer = nullptr;
            o.d_workspace = nullptr; o.stream = nullptr;
        }
        return *this;
    }

    void run() {
        T alpha = static_cast<T>(1), beta = static_cast<T>(0);
        detail_bs::check_cutensor(
            cutensorBlockSparseContract(
                handle, plan,
                &alpha, a_desc.d_block_ptrs.data(),
                        b_desc.d_block_ptrs.data(),
                &beta,  d_C_ptrs.data(),
                        d_D_ptrs.data(),
                d_workspace, ws_size, stream),
            "cutensorBlockSparseContract");
        cudaStreamSynchronize(stream);
    }
};

// Prepare a cuTENSOR block-sparse contraction for repeated execution.
// All setup (descriptors, plan, workspace, device buffers) is done once.
// Call .run() to execute only cutensorBlockSparseContract + sync.
// Only supports non-batched contractions.
template <typename T>
CutensorPreparedContract<T> prepare_cutensor_block_sparse(
    const BlockSparseTensor& A_bst,
    const BlockSparseTensor& B_bst,
    const EinsumExpression& expr,
    const std::vector<std::vector<int64_t>>* precomputed_output_coords = nullptr)
{
    CutensorPreparedContract<T> pc;
    pc.handle = get_cached_cutensor_handle();

    pc.a_desc = detail_bs::create_bs_descriptor<T>(pc.handle, A_bst);
    pc.b_desc = detail_bs::create_bs_descriptor<T>(pc.handle, B_bst);

    auto output_coords = precomputed_output_coords
        ? *precomputed_output_coords
        : compute_output_block_coords(A_bst, B_bst, expr);
    if (output_coords.empty()) return pc;

    auto out_bst = build_output_bst_meta(A_bst, B_bst, expr, output_coords);
    const uint64_t out_nnz = static_cast<uint64_t>(output_coords.size());

    std::vector<int64_t> out_extents;
    std::vector<uint32_t> out_sections;
    detail_bs::build_extent_arrays(out_bst, out_extents, out_sections);
    auto out_flat_coords = detail_bs::flatten_block_coords(out_bst);

    detail_bs::check_cutensor(
        cutensorCreateBlockSparseTensorDescriptor(
            pc.handle, &pc.out_desc,
            static_cast<uint32_t>(out_bst.meta.ndim), out_nnz,
            out_sections.data(), out_extents.data(),
            out_flat_coords.data(), nullptr,
            to_cutensor_type(out_bst.meta.dtype)),
        "cutensorCreateBlockSparseTensorDescriptor (output)");

    std::vector<int64_t> out_offsets;
    auto [out_block_volumes, out_total_elements] =
        compute_block_volumes_and_offsets(out_bst, output_coords, out_offsets);
    if (out_total_elements > 0) {
        cudaMalloc(&pc.d_out_buffer, out_total_elements * sizeof(T));
        cudaMemset(pc.d_out_buffer, 0, out_total_elements * sizeof(T));
    }

    pc.d_C_ptrs.resize(out_nnz);
    pc.d_D_ptrs.resize(out_nnz);
    int64_t ptr_offset = 0;
    for (uint64_t i = 0; i < out_nnz; ++i) {
        T* ptr = static_cast<T*>(pc.d_out_buffer) + ptr_offset;
        pc.d_C_ptrs[i] = static_cast<const void*>(ptr);
        pc.d_D_ptrs[i] = static_cast<void*>(ptr);
        ptr_offset += out_block_volumes[i];
    }

    std::vector<int32_t> mA, mB, mO;
    for (char c : A_bst.meta.indices) mA.push_back(static_cast<int32_t>(c));
    for (char c : B_bst.meta.indices) mB.push_back(static_cast<int32_t>(c));
    for (char c : expr.output_labels) mO.push_back(static_cast<int32_t>(c));

    auto compute_desc = to_cutensor_compute(A_bst.meta.dtype);

    cutensorOperationDescriptor_t opDesc = nullptr;
    detail_bs::check_cutensor(
        cutensorCreateBlockSparseContraction(
            pc.handle, &opDesc,
            pc.a_desc.desc, mA.data(), CUTENSOR_OP_IDENTITY,
            pc.b_desc.desc, mB.data(), CUTENSOR_OP_IDENTITY,
            pc.out_desc, mO.data(), CUTENSOR_OP_IDENTITY,
            pc.out_desc, mO.data(),
            compute_desc),
        "cutensorCreateBlockSparseContraction");

    cutensorPlanPreference_t pref = nullptr;
    cutensorCreatePlanPreference(pc.handle, &pref,
                                 CUTENSOR_ALGO_DEFAULT,
                                 CUTENSOR_JIT_MODE_NONE);

    cutensorEstimateWorkspaceSize(pc.handle, opDesc, pref,
                                  CUTENSOR_WORKSPACE_DEFAULT, &pc.ws_size);

    detail_bs::check_cutensor(
        cutensorCreatePlan(pc.handle, &pc.plan, opDesc, pref, pc.ws_size),
        "cutensorCreatePlan (block sparse)");

    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(opDesc);

    if (pc.ws_size > 0) cudaMalloc(&pc.d_workspace, pc.ws_size);

    cudaStreamCreate(&pc.stream);

    return pc;
}

// Main entry point: run cuTENSOR block-sparse contraction.
// Handles batch modes by looping over batch slices.
// If d_A_values / d_B_values are non-null, uses those device buffers
// instead of uploading (only for the non-batch path; ignored for batch slicing).
template <typename T>
CutensorBlockSparseResult<T> run_cutensor_block_sparse(
    const BlockSparseTensor& A_bst,
    const BlockSparseTensor& B_bst,
    const EinsumExpression& expr,
    const T* d_A_values = nullptr,
    const T* d_B_values = nullptr,
    const std::vector<std::vector<int64_t>>* precomputed_output_coords = nullptr)
{
    // Check for batch indices
    EinsumIndexSets index_sets;
    create_einsum_index_sets(expr, index_sets);

    if (index_sets.batch_indices.empty()) {
        // No batch modes — direct call
        return run_cutensor_block_sparse_single<T>(A_bst, B_bst, expr,
                                                    d_A_values, d_B_values,
                                                    precomputed_output_coords);
    }

    // --- Batch mode workaround ---
    // cuTENSOR block-sparse beta doesn't support modes in all 3 operands.
    // Loop over unique batch coordinate combinations, running per-slice
    // contractions with batch dims removed.

    // Find batch dim positions in A, B, and output
    std::vector<int> batch_dims_A, batch_dims_B, batch_dims_out;
    for (char c : index_sets.batch_indices) {
        batch_dims_A.push_back(label_pos(A_bst.meta.indices, c));
        batch_dims_B.push_back(label_pos(B_bst.meta.indices, c));
        batch_dims_out.push_back(label_pos(
            std::vector<char>(expr.output_labels.begin(), expr.output_labels.end()), c));
    }

    // Build the slice einsum expression (output labels without batch dims)
    std::string slice_output;
    for (size_t i = 0; i < expr.output_labels.size(); ++i) {
        bool is_batch = false;
        for (int bd : batch_dims_out) {
            if (static_cast<int>(i) == bd) { is_batch = true; break; }
        }
        if (!is_batch) slice_output += expr.output_labels[i];
    }

    // Build slice einsum: keep all input labels (batch dims become free in the slice,
    // but since they'll have only 1 section in the slice, they'll be contracted properly).
    // Actually, for the slice we need to remove batch labels from input labels too.
    std::string slice_A_labels, slice_B_labels;
    for (size_t i = 0; i < A_bst.meta.indices.size(); ++i) {
        bool is_batch = false;
        for (int bd : batch_dims_A) {
            if (static_cast<int>(i) == bd) { is_batch = true; break; }
        }
        if (!is_batch) slice_A_labels += A_bst.meta.indices[i];
    }
    for (size_t i = 0; i < B_bst.meta.indices.size(); ++i) {
        bool is_batch = false;
        for (int bd : batch_dims_B) {
            if (static_cast<int>(i) == bd) { is_batch = true; break; }
        }
        if (!is_batch) slice_B_labels += B_bst.meta.indices[i];
    }

    std::string slice_einsum_str = slice_A_labels + "," + slice_B_labels + "->" + slice_output;
    auto slice_expr = parse_einsum(slice_einsum_str);

    // Build full output metadata for the combined dense result
    auto full_output_coords = compute_output_block_coords(A_bst, B_bst, expr);
    auto full_out_bst = build_output_bst_meta(A_bst, B_bst, expr, full_output_coords);

    CutensorBlockSparseResult<T> result;
    result.kernel_ms = 0.0f;

    // Prepare block-sparse output: zero-initialized per-block buffers
    std::vector<int64_t> full_offsets;
    auto [full_volumes, full_total] =
        compute_block_volumes_and_offsets(full_out_bst, full_output_coords, full_offsets);
    result.block_coords = full_output_coords;
    result.block_volumes = full_volumes;
    result.block_offsets = full_offsets;
    result.total_block_elements = full_total;
    result.block_data.assign(static_cast<size_t>(full_total), static_cast<T>(0));

    // Build coord → block index map for the full output
    std::map<std::vector<int64_t>, size_t> full_coord_to_idx;
    for (size_t i = 0; i < full_output_coords.size(); ++i)
        full_coord_to_idx[full_output_coords[i]] = i;

    // Build mapping from slice dim -> full output dim (precompute once)
    std::vector<int> slice_to_full;
    for (size_t i = 0; i < expr.output_labels.size(); ++i) {
        bool is_batch = false;
        for (int bd : batch_dims_out) {
            if (static_cast<int>(i) == bd) { is_batch = true; break; }
        }
        if (!is_batch) slice_to_full.push_back(static_cast<int>(i));
    }

    // Enumerate unique batch section coordinates
    std::set<std::vector<int64_t>> batch_section_set;
    for (const auto& coord : A_bst.block_coords) {
        std::vector<int64_t> bc;
        for (int d : batch_dims_A) bc.push_back(coord[d]);
        batch_section_set.insert(bc);
    }
    for (const auto& coord : B_bst.block_coords) {
        std::vector<int64_t> bc;
        for (int d : batch_dims_B) bc.push_back(coord[d]);
        batch_section_set.insert(bc);
    }

    // For each batch section, loop over element values within the section
    for (const auto& batch_sec : batch_section_set) {
        // Get batch section extents (should be same across all tensors for matching modes)
        std::vector<int64_t> batch_extents;
        for (size_t bi = 0; bi < batch_dims_A.size(); ++bi) {
            char lbl = A_bst.meta.indices[batch_dims_A[bi]];
            batch_extents.push_back(
                A_bst.block_sizes.at(lbl)[static_cast<size_t>(batch_sec[bi])]);
        }

        // Total number of element-level values in this batch section
        int64_t batch_volume = 1;
        for (int64_t e : batch_extents) batch_volume *= e;

        for (int64_t b_elem = 0; b_elem < batch_volume; ++b_elem) {
            // Decompose b_elem into per-batch-dim local indices
            std::vector<int64_t> batch_local(batch_extents.size());
            int64_t rem = b_elem;
            for (int bi = static_cast<int>(batch_extents.size()) - 1; bi >= 0; --bi) {
                batch_local[bi] = rem % batch_extents[bi];
                rem /= batch_extents[bi];
            }

            // Extract element-level slices
            auto A_slice = detail_bs::slice_bst_for_batch<T>(
                A_bst, batch_dims_A, batch_sec, batch_local);
            auto B_slice = detail_bs::slice_bst_for_batch<T>(
                B_bst, batch_dims_B, batch_sec, batch_local);

            if (A_slice.block_coords.empty() || B_slice.block_coords.empty())
                continue;

            std::vector<std::vector<int64_t>> slice_shapes = {
                block_utils::compute_full_shape(A_slice),
                block_utils::compute_full_shape(B_slice)
            };
            auto local_expr = slice_expr;
            validate_einsum(local_expr, slice_shapes);

            auto slice_result = run_cutensor_block_sparse_single<T>(
                A_slice, B_slice, local_expr);
            result.kernel_ms += slice_result.kernel_ms;
            result.setup_ms += slice_result.setup_ms;
            if (slice_result.peak_gpu_bytes > result.peak_gpu_bytes)
                result.peak_gpu_bytes = slice_result.peak_gpu_bytes;

            // Place slice block-sparse data into full block-sparse output
            int full_rank = static_cast<int>(expr.output_labels.size());
            int slice_rank = static_cast<int>(slice_result.block_coords.empty()
                ? 0 : slice_result.block_coords[0].size());

            for (size_t sblk = 0; sblk < slice_result.block_coords.size(); ++sblk) {
                // Build full coord by inserting batch section coords
                std::vector<int64_t> full_coord(full_rank);
                for (size_t bi = 0; bi < batch_dims_out.size(); ++bi)
                    full_coord[batch_dims_out[bi]] = batch_sec[bi];
                for (int sd = 0; sd < slice_rank; ++sd)
                    full_coord[slice_to_full[sd]] = slice_result.block_coords[sblk][sd];

                auto it = full_coord_to_idx.find(full_coord);
                if (it == full_coord_to_idx.end()) continue;
                size_t fidx = it->second;

                // Get full block shape and compute col-major strides
                auto full_shape = block_utils::block_shape_for_coord(
                    full_out_bst, full_output_coords[fidx]);
                std::vector<int64_t> full_cm(full_rank);
                if (full_rank > 0) {
                    full_cm[0] = 1;
                    for (int d = 1; d < full_rank; ++d)
                        full_cm[d] = full_cm[d - 1] * full_shape[d - 1];
                }

                // Compute batch element offset in the full block (col-major)
                int64_t batch_elem_offset = 0;
                for (size_t bi = 0; bi < batch_dims_out.size(); ++bi)
                    batch_elem_offset += batch_local[bi] * full_cm[batch_dims_out[bi]];

                // Get slice block shape and col-major strides
                int64_t slice_vol = slice_result.block_volumes[sblk];
                std::vector<int64_t> slice_shape;
                for (int d = 0; d < full_rank; ++d) {
                    bool is_bd = false;
                    for (int bd : batch_dims_out) {
                        if (d == bd) { is_bd = true; break; }
                    }
                    if (!is_bd) slice_shape.push_back(full_shape[d]);
                }
                int srank = static_cast<int>(slice_shape.size());
                std::vector<int64_t> slice_cm(srank);
                if (srank > 0) {
                    slice_cm[0] = 1;
                    for (int d = 1; d < srank; ++d)
                        slice_cm[d] = slice_cm[d - 1] * slice_shape[d - 1];
                }

                // Copy elements: decompose slice col-major index → remap to full col-major
                const T* src = slice_result.block_data.data() + slice_result.block_offsets[sblk];
                T* dst = result.block_data.data() + result.block_offsets[fidx];
                for (int64_t elem = 0; elem < slice_vol; ++elem) {
                    int64_t remaining = elem;
                    int64_t full_off = batch_elem_offset;
                    for (int d = srank - 1; d >= 0; --d) {
                        int64_t idx = remaining / slice_cm[d];
                        remaining -= idx * slice_cm[d];
                        full_off += idx * full_cm[slice_to_full[d]];
                    }
                    dst[full_off] = src[elem];
                }
            }
        }
    }

    return result;
}

} // namespace bsptc
