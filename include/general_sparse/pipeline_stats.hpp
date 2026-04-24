#pragma once

// Timing/diagnostic statistics for the pipeline.
// Separated from pipeline.cuh to keep the main pipeline logic focused.
// NOTE: This header is included from within namespace bsptc in pipeline.cuh,
// after PipelineTimingBreakdown and PipelinePlan are defined.
// All required types and includes are already available at the include point.

// Populate timing breakdown stats: FLOPs, output elems, transpose bytes, detail strings.
template <typename T>
void populate_timing_stats(
    PipelineTimingBreakdown& t,
    const PipelinePlan<T>& plan,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu)
{
    auto stats = compute_partition_stats(plan.work);
    t.large_gemm_problems = static_cast<int>(stats.num_gemm_problems);
    t.medium_descs = plan.medium_vol_num_problems;
    t.medium_shapes = plan.medium_vol_num_shapes;
    t.small_descs = plan.small_num_descs;
    t.small_shapes = plan.small_num_shapes;
    t.small_warps = plan.total_warps;
    t.large_output_elems = stats.large_output_elements;
    t.medium_output_elems = stats.medium_output_elements;
    t.small_output_elems = stats.small_output_elements;

    // FLOPs per tier: 2*M*N*K*batch for each problem
    auto sum_flops = [](const std::vector<GemmProblemDesc>& probs) {
        int64_t f = 0;
        for (const auto& p : probs) f += 2LL * p.M * p.N * p.K * p.batch_size;
        return f;
    };
    t.large_flops = sum_flops(plan.work.large.problems);
    t.medium_flops = sum_flops(plan.work.medium.problems);
    t.small_flops = sum_flops(plan.work.small.problems);

    // Build large-tier tile info string (non-streaming: from nonstream_classified,
    // streaming: already printed per-queue in streaming section below).
    if (!plan.streaming.enabled && !plan.work.large.empty()) {
        std::string info;
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            const auto& qwork = plan.nonstream_classified.queues[qi];
            if (qwork.empty()) continue;
            TileShape tile = best_tile_for_queue(qwork);
            if (!info.empty()) info += " ";
            info += std::string(shape_queue_name(static_cast<ShapeQueue>(qi)))
                  + "=" + tile_shape_name(tile);
        }
        t.large_tile_info = std::move(info);
    }

    // Transpose bytes
    bool will_transpose = plan.needs_transpose;
    int64_t transpose_bytes = 0;
    if (will_transpose) {
        if (plan.layout_plan.plan_A.transpose)
            transpose_bytes += A_dev.total_elements * static_cast<int64_t>(sizeof(T));
        if (plan.layout_plan.plan_B.transpose)
            transpose_bytes += B_dev.total_elements * static_cast<int64_t>(sizeof(T));
        transpose_bytes *= 2;  // read + write
    }
    t.transpose_bytes = transpose_bytes;

    // Transpose dispatch kind strings
    auto kind_str = [](const GemmLayoutPlan::OperandPlan& op) -> const char* {
        if (!op.transpose) return nullptr;
        if (op.dispatch_kind == TransposeDispatchKind::GenericPermute) return "Hypertile";
        return nullptr;
    };
    t.transpose_A_kind = will_transpose ? kind_str(plan.layout_plan.plan_A) : nullptr;
    t.transpose_B_kind = will_transpose ? kind_str(plan.layout_plan.plan_B) : nullptr;

    // Detail strings: perm + key dimension ranges
    auto make_detail = [](const GemmLayoutPlan::OperandPlan& op,
                          const GPUSparseInput<T>& gpu,
                          const TransposeFlatMeta& meta) -> std::string {
        if (!op.transpose) return {};
        std::string s = "perm=[";
        for (size_t i = 0; i < op.perm.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(op.perm[i]);
        }
        s += "]";
        auto range_str = [](int64_t lo, int64_t hi) {
            if (lo == hi) return std::to_string(lo);
            return std::to_string(lo) + "-" + std::to_string(hi);
        };
        if (op.dispatch_kind == TransposeDispatchKind::GenericPermute) {
            dim_t rank = gpu.rank;
            dim_t head_rank = rank - op.unmoved_tail_dims;
            int64_t min_tail = INT64_MAX, max_tail = 0;
            for (int b = 0; b < static_cast<int>(gpu.num_blocks); ++b) {
                int64_t tail = 1;
                for (int32_t d = head_rank; d < rank; ++d)
                    tail *= gpu.block_dims[b * rank + d];
                min_tail = std::min(min_tail, tail);
                max_tail = std::max(max_tail, tail);
            }
            s += " tail=" + range_str(min_tail, max_tail)
               + " head_rank=" + std::to_string(head_rank);
            if (meta.ht_num_slices > 0) {
                int ht_total_tasks = 0;
                for (int i = 0; i < meta.ht_num_slices; ++i) ht_total_tasks += meta.ht_slices[i].num_tasks;
                s += " ht_tasks=" + std::to_string(ht_total_tasks)
                   + " ht_tiles=" + std::to_string(meta.ht_total_tiles)
                   + " sq_rank=" + std::to_string(meta.ht_squashed_rank)
                   + " RR=" + std::to_string(meta.ht_min_rr)
                   + " WR=" + std::to_string(meta.ht_min_wr)
                   + " tile_vol=" + std::to_string(meta.ht_tile_volume);
            }
        }
        return s;
    };
    if (will_transpose) {
        t.transpose_A_detail = make_detail(plan.layout_plan.plan_A, A_gpu, plan.transpose_precomp_A);
        t.transpose_B_detail = make_detail(plan.layout_plan.plan_B, B_gpu, plan.transpose_precomp_B);
    }

    // Peak GPU memory (bytes): measured via cudaMemGetInfo in pipeline_allocate,
    // plus caller-owned input values.  Falls back to estimate when measurement
    // is unavailable (e.g. external workspace).
    const int64_t elem = static_cast<int64_t>(sizeof(T));
    int64_t input_bytes = (A_dev.total_elements + B_dev.total_elements) * elem;
    if (plan.measured_alloc_bytes > 0) {
        t.peak_gpu_bytes = input_bytes + plan.measured_alloc_bytes;
    } else {
        // Fallback estimate: sum of all pipeline-owned allocations
        int64_t peak = input_bytes;
        if (plan.streaming.enabled && will_transpose && plan.streaming.buf_cap > 0) {
            int total_batches = 0;
            for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi)
                total_batches += static_cast<int>(plan.streaming.queue_partitions[qi].batches.size());
            int num_bufs = (total_batches <= 1) ? 1 : 2;
            peak += num_bufs * plan.streaming.buf_cap * elem;
            for (int s = 0; s < num_bufs; ++s)
                for (int op = 0; op < 2; ++op)
                    peak += plan.streaming.precomp_bufs[s][op].total_bytes();
        } else if (will_transpose) {
            if (plan.layout_plan.plan_A.transpose)
                peak += A_dev.total_elements * elem;
            if (plan.layout_plan.plan_B.transpose)
                peak += B_dev.total_elements * elem;
        }
        peak += plan.total_pair_output_size * elem;
        peak += plan.accum_map.total_output_elements * elem;
        t.peak_gpu_bytes = peak;
    }

    // Streaming info
    t.streaming_enabled = plan.streaming.enabled;
    if (plan.streaming.enabled) {
        t.streaming_buf_cap = plan.streaming.buf_cap;
        int total = 0;
        for (int qi = 0; qi < NUM_SHAPE_QUEUES; ++qi) {
            auto& qi_info = t.streaming_queues[qi];
            qi_info.batches = static_cast<int>(plan.streaming.queue_partitions[qi].batches.size());
            qi_info.problems = static_cast<int>(plan.streaming.queue_work[qi].problems.size());
            qi_info.tile_name = tile_shape_name(plan.streaming.queue_tiles[qi]);
            total += qi_info.batches;
        }
        t.streaming_total_batches = total;

        // Pre-allocated transpose descriptor buffer sizes
        int64_t precomp_total = 0;
        for (int s = 0; s < 2; ++s)
            for (int op = 0; op < 2; ++op)
                precomp_total += plan.streaming.precomp_bufs[s][op].total_bytes();
        t.streaming_precomp_bytes = precomp_total;
    }
}
