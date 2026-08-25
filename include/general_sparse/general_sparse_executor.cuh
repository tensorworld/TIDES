#pragma once

// Pipeline two-phase API (setup + execute + extract) — canonical location for
// GeneralSparseResult, PipelineTimingBreakdown, DispatchConfig, TransposedValues,
// ensure_gemm_layout, PipelinePlan, pipeline_setup/execute/extract_result.
#include "pipeline.cuh"

namespace bsptc {

// One-call pipeline: setup + execute + extract.
// Caller manages A_dev/B_dev lifetime (must stay alive until this returns).
template <typename T>
GeneralSparseResult<T> execute_general_sparse_pipeline(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    const EinsumIndexSets& index_sets,
    const std::string& output_labels = "",
    const DispatchConfig& dispatch_config = DispatchConfig(),
    PipelineTimingBreakdown* timing = nullptr)
{
    PipelinePlan<T> plan;
    if (!pipeline_setup<T>(plan, A_gpu, B_gpu, A_dev, B_dev,
                           index_sets, output_labels, dispatch_config, timing))
        throw std::runtime_error(
            "execute_general_sparse_pipeline: pipeline_setup failed (see stderr)");

    pipeline_execute<T>(plan, A_gpu, B_gpu, A_dev, B_dev);

    return pipeline_extract_result(plan);
}

#ifndef BSPTC_PIPELINE_INSTANTIATE
extern template GeneralSparseResult<float> execute_general_sparse_pipeline<float>(const GPUSparseInput<float>&, const GPUSparseInput<float>&, const DeviceSparseInput<float>&, const DeviceSparseInput<float>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*);
extern template GeneralSparseResult<double> execute_general_sparse_pipeline<double>(const GPUSparseInput<double>&, const GPUSparseInput<double>&, const DeviceSparseInput<double>&, const DeviceSparseInput<double>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*);
#endif

} // namespace bsptc
