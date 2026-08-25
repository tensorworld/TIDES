#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Stage 1: Launch transpose kernels (async on stream 0 + plan.stream_large).
// Records plan.transpose_done_flat and plan.transpose_done_tiled events.
template <typename T>
void stage_transpose(PipelinePlan<T>& plan,
                     const DeviceSparseInput<T>& A_dev, const DeviceSparseInput<T>& B_dev)
{
    if (!plan.needs_transpose) return;

    plan.tv = ensure_gemm_layout(A_dev, B_dev, plan.layout_plan,
                                 plan.transpose_precomp_A, plan.transpose_precomp_B,
                                 plan.tv.d_workspace_A, plan.tv.d_workspace_B,
                                 plan.stream_large);
    cudaEventRecord(plan.transpose_done_flat, 0);
    cudaEventRecord(plan.transpose_done_tiled, plan.stream_large);
}

// Stage 2: Patch descriptor pointers to use transposed data.
// Call after transpose events have been recorded (not necessarily completed on device —
// the H2D copies are enqueued on the per-tier streams which already wait on the events).
template <typename T>
void stage_patch_pointers(PipelinePlan<T>& plan,
                          const DeviceSparseInput<T>& A_dev, const DeviceSparseInput<T>& B_dev)
{
    if (!plan.needs_transpose) return;
    if (plan.tv.d_A == A_dev.d_values && plan.tv.d_B == B_dev.d_values) return;

    const auto& triv_A = plan.trivial_shapes_A;
    const auto& triv_B = plan.trivial_shapes_B;
    const auto& smap_A = plan.shape_map_A;
    const auto& smap_B = plan.shape_map_B;

    // Only large tier (CUTLASS) needs pointer patching after transpose.
    // Volume and warp kernels handle interleaved layouts natively.
    // Patch per-queue runner pinned buffers and re-upload to device.
    if (!plan.work.large.empty()) {
        const bool has_triv_A = !triv_A.empty();
        const bool has_triv_B = !triv_B.empty();
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            auto& runner = plan.nonstream_runners[qi];
            if (!runner) continue;
            auto& qwork = plan.nonstream_classified.queues[qi];
            T** h_pA = static_cast<T**>(runner.h_ptr_A_loc);
            T** h_pB = static_cast<T**>(runner.h_ptr_B_loc);
            size_t idx = 0;
            for (const auto& prob : qwork.problems) {
                T* A_base = (has_triv_A && triv_A[smap_A.block_to_shape[prob.block_idx_A]])
                    ? A_dev.d_values : plan.tv.d_A;
                T* B_base = (has_triv_B && triv_B[smap_B.block_to_shape[prob.block_idx_B]])
                    ? B_dev.d_values : plan.tv.d_B;
                int64_t A_batch_stride = static_cast<int64_t>(prob.M) * prob.K;
                int64_t B_batch_stride = static_cast<int64_t>(prob.K) * prob.N;
                for (int64_t b = 0; b < prob.batch_size; ++b) {
                    h_pA[idx] = A_base + prob.offset_A + b * A_batch_stride;
                    h_pB[idx] = B_base + prob.offset_B + b * B_batch_stride;
                    ++idx;
                }
            }
            runner.patch_ab_pointers(plan.stream_large);
        }
    }
}

// Stage 3: Launch large GEMM (CUTLASS grouped).
// Non-streaming: run each per-queue pre-prepared runner sequentially.
template <typename T>
void stage_large_gemm(PipelinePlan<T>& plan, cudaStream_t stream)
{
    for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
        if (plan.nonstream_runners[qi])
            plan.nonstream_runners[qi].run(stream);
    }
}

// Stage 4: Launch medium GEMM (volume kernel).
template <typename T>
void stage_medium_gemm(PipelinePlan<T>& plan, cudaStream_t stream)
{
    if (plan.medium_vol_num_problems > 0) {
        launch_volume_gemm_dispatch<T>(plan.d_medium_vol_shapes,
            plan.d_medium_vol_mems,
            plan.medium_vol_num_problems,
            plan.medium_vol_max_smem_elems,
            plan.medium_vol_max_tiles, stream);
    }
}

// Stage 5: Launch small GEMM (warp kernel) + dot-reduce for scalar descs.
template <typename T>
void stage_small_gemm(PipelinePlan<T>& plan, cudaStream_t stream)
{
    if (plan.total_warps > 0) {
        launch_warp_gemm<T>(plan.d_small_shapes, plan.d_small_mems,
            plan.d_warp_prefix, plan.small_num_descs,
            plan.total_warps, plan.warp_variant, stream);
    }
    if (plan.small_num_dots > 0) {
        launch_dot_reduce<T>(plan.d_small_shapes, plan.d_small_mems,
            plan.d_dot_indices, plan.small_num_dots,
            plan.warp_variant, stream);
    }
}

// Stage 3b: Per-queue streamed large GEMM.
// Each shape queue (Square, SkinnyTall, SkinnyWide) runs sequentially,
// using a fixed tile matched to its shape class. Double-buffering continues
// within each queue; between queues, host-sync both slots for clean handoff.
template <typename T>
void stage_large_gemm_streamed_perqueue(PipelinePlan<T>& plan,
                                const GPUSparseInput<T>& A_gpu, const GPUSparseInput<T>& B_gpu,
                                const DeviceSparseInput<T>& A_dev, const DeviceSparseInput<T>& B_dev)
{
    auto& st = plan.streaming;
    if (!st.enabled) return;

    // Fast path: GEMM args + transpose metadata precomputed during setup.
    // Only transpose kernel launch and GEMM launch remain.
    // No sync here — outer cudaDeviceSynchronize handles it.
    if (st.precomputed) {
        if (!plan.needs_transpose) {
            // No transpose: single stream, no events.
            cudaStream_t s = st.streams[0];
            for (int i = 0; i < st.num_precomp; ++i) {
                if (st.args[i])
                    st.args[i].run(s);
            }
            return;
        }

        // Transpose: double-buffered streams with overlap.
        // T_i launches first (writes d_buf[slot], no cross-stream conflict).
        // G_i waits for G_{i-1} (shared d_output).  Same-stream ordering
        // protects d_buf reuse (T_{i+2} after G_i on the same stream).
        for (int i = 0; i < st.num_precomp; ++i) {
            int slot = i % 2;
            cudaStream_t s = st.streams[slot];
            T* d_buf = st.d_buf[slot];

            const auto& meta_A = st.precomp_meta[slot][0];
            const auto& meta_B = st.precomp_meta[slot][1];
            if (plan.layout_plan.plan_A.transpose && meta_A.valid) {
                launch_batched_transpose<T>(A_dev.d_values, d_buf, meta_A, s);
            }
            if (plan.layout_plan.plan_B.transpose && meta_B.valid) {
                launch_batched_transpose<T>(B_dev.d_values, d_buf, meta_B, s);
            }

            if (i > 0) {
                cudaStreamWaitEvent(s, st.gemm_done[1 - slot]);
            }
            if (st.args[slot]) {
                st.args[slot].run(s);
            }
            cudaEventRecord(st.gemm_done[slot], s);
        }
        return;
    }

    // Fallback: runtime batch construction (total batches > 2)
    bool slot_active[2] = {false, false};
    std::vector<TransposeFlatMeta> deferred_precomps;

    for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
        const auto& partition = st.queue_partitions[qi];
        const auto& queue_large = st.queue_work[qi];
        KernelChoice kernel = st.queue_kernels[qi];

        if (partition.batches.empty() || queue_large.empty()) continue;

        // Precompute per-problem expanded index within this queue's work
        std::vector<int64_t> expanded_start(queue_large.problems.size());
        int64_t idx = 0;
        for (size_t j = 0; j < queue_large.problems.size(); ++j) {
            expanded_start[j] = idx;
            idx += queue_large.problems[j].batch_size;
        }

        for (size_t bi = 0; bi < partition.batches.size(); ++bi) {
            const auto& batch = partition.batches[bi];
            if (batch.problem_indices.empty()) continue;

            int slot = static_cast<int>(bi % 2);
            cudaStream_t s = st.streams[slot];
            T* d_buf = st.d_buf[slot];

            // Wait for previous iteration on this slot before reusing buffers.
            if (slot_active[slot]) {
                // GPU-side: new transpose must not start until old GEMM finishes reading d_buf
                cudaStreamWaitEvent(st.streams[slot], st.gemm_done[slot]);
                // Host-side: wait for old H2D copy to finish before CPU overwrites h_args_buf
                cudaEventSynchronize(st.copy_done[slot]);
            }

            // 1. Build per-batch transpose precomp (using pre-allocated buffers)
            TransposeFlatMeta precomp_A, precomp_B;
            if (plan.layout_plan.plan_A.transpose) {
                precomp_A = assemble_hypertile_batch(
                    batch.unique_blocks_A,
                    A_gpu.block_offsets.data(), A_gpu.block_volumes.data(),
                    plan.shape_map_A,
                    plan.hypertile_index_A, plan.trivial_shapes_A,
                    batch.pack_offset_A, &st.precomp_bufs[slot][0], s);
            }
            if (plan.layout_plan.plan_B.transpose) {
                precomp_B = assemble_hypertile_batch(
                    batch.unique_blocks_B,
                    B_gpu.block_offsets.data(), B_gpu.block_volumes.data(),
                    plan.shape_map_B,
                    plan.hypertile_index_B, plan.trivial_shapes_B,
                    batch.pack_offset_B, &st.precomp_bufs[slot][1], s);
            }

            // 2. Launch transpose on this slot's stream
            if (plan.layout_plan.plan_A.transpose && precomp_A.kind != TransposeDispatchKind::None) {
                launch_batched_transpose<T>(A_dev.d_values, d_buf, precomp_A, s);
            }
            if (plan.layout_plan.plan_B.transpose && precomp_B.kind != TransposeDispatchKind::None) {
                launch_batched_transpose<T>(B_dev.d_values, d_buf, precomp_B, s);
            }

            // 3. Build per-batch LargeBlockWork with pointers into d_buf
            auto batch_work = build_streaming_batch_work(
                plan, queue_large, batch, expanded_start,
                precomp_A, precomp_B, d_buf, A_dev.d_values, B_dev.d_values);

            // 4. Wait for other slot and launch GEMM with fixed tile, beta=1
            int other = 1 - slot;
            if (slot_active[other]) {
                cudaStreamWaitEvent(s, st.gemm_done[other]);
            }
            fill_large_args_dispatch(st.args[slot], batch_work, T(1), T(1), kernel, s);
            cudaEventRecord(st.copy_done[slot], s);  // fires after H2D, before GEMM
            if (st.args[slot]) {
                st.args[slot].run(s);
            }
            cudaEventRecord(st.gemm_done[slot], s);
            slot_active[slot] = true;
            deferred_precomps.push_back(precomp_A);
            deferred_precomps.push_back(precomp_B);
        }

        // Between queues: wait for both slots to complete for clean handoff
        if (slot_active[0]) cudaEventSynchronize(st.gemm_done[0]);
        if (slot_active[1]) cudaEventSynchronize(st.gemm_done[1]);
        slot_active[0] = slot_active[1] = false;
    }

    // Cleanup — between-queue syncs already ensured GPU completion
    for (auto& p : deferred_precomps) p.free_gpu();
}

// Stage 6: Launch accumulation (reduce_noatomic from temp buffer → d_output).
template <typename T>
void stage_accumulate(PipelinePlan<T>& plan, cudaStream_t stream)
{
    if (plan.num_pairs > 0 && plan.accum_map.total_output_elements > 0) {
        launch_reduce_noatomic<T, 256, 4>(
            plan.d_temp_buffer, plan.d_output, plan.reduce_plan, stream);
    }
}

