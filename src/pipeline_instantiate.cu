// Explicit instantiation of pipeline template functions for float.
// Compiled once as a separate translation unit — other TUs use extern template
// declarations to avoid re-instantiating these heavy templates on every rebuild.

#define BSPTC_PIPELINE_INSTANTIATE
#include "general_sparse/general_sparse_executor.cuh"

namespace bsptc {

// --- pipeline.cuh functions ---
template void launch_batched_transpose<float>(const float*, float*, const TransposeFlatMeta&, cudaStream_t);
template TransposedValues<float> ensure_gemm_layout<float>(const DeviceSparseInput<float>&, const DeviceSparseInput<float>&, const GemmLayoutPlan&, const TransposeFlatMeta&, const TransposeFlatMeta&, float*, float*, cudaStream_t);
template void free_transposed_values<float>(TransposedValues<float>&);
template bool prepare_small_shapes<float>(PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&);
template void bind_small_mems<float>(PipelinePlan<float>&, const DeviceSparseInput<float>&, const DeviceSparseInput<float>&);
template bool pipeline_plan<float>(PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*, const OutputBlockMap*);
template PipelineWorkspaceInfo pipeline_query_workspace<float>(const PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&);
template bool pipeline_prepare<float>(PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&);
template bool pipeline_allocate<float>(PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&, const DeviceSparseInput<float>&, const DeviceSparseInput<float>&, float*, void*, int64_t);
template bool pipeline_setup<float>(PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&, const DeviceSparseInput<float>&, const DeviceSparseInput<float>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*, const OutputBlockMap*);
template void pipeline_execute<float>(PipelinePlan<float>&, const GPUSparseInput<float>&, const GPUSparseInput<float>&, const DeviceSparseInput<float>&, const DeviceSparseInput<float>&);
template GeneralSparseResult<float> pipeline_extract_result<float>(PipelinePlan<float>&);

// --- general_sparse_executor.cuh functions ---
template GeneralSparseResult<float> execute_general_sparse_pipeline<float>(const GPUSparseInput<float>&, const GPUSparseInput<float>&, const DeviceSparseInput<float>&, const DeviceSparseInput<float>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*);

} // namespace bsptc
