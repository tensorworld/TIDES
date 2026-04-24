// Explicit instantiation of pipeline template functions for double.
// See pipeline_instantiate.cu (float) for details.

#define BSPTC_PIPELINE_INSTANTIATE
#include "general_sparse/general_sparse_executor.cuh"

namespace bsptc {

// --- pipeline.cuh functions ---
template void launch_batched_transpose<double>(const double*, double*, const TransposeFlatMeta&, cudaStream_t);
template TransposedValues<double> ensure_gemm_layout<double>(const DeviceSparseInput<double>&, const DeviceSparseInput<double>&, const GemmLayoutPlan&, const TransposeFlatMeta&, const TransposeFlatMeta&, double*, double*, cudaStream_t);
template void free_transposed_values<double>(TransposedValues<double>&);
template void prepare_small_shapes<double>(PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&);
template void bind_small_mems<double>(PipelinePlan<double>&, const DeviceSparseInput<double>&, const DeviceSparseInput<double>&);
template bool pipeline_plan<double>(PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*, const OutputBlockMap*);
template PipelineWorkspaceInfo pipeline_query_workspace<double>(const PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&);
template void pipeline_prepare<double>(PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&);
template bool pipeline_allocate<double>(PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&, const DeviceSparseInput<double>&, const DeviceSparseInput<double>&, double*, void*, int64_t);
template bool pipeline_setup<double>(PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&, const DeviceSparseInput<double>&, const DeviceSparseInput<double>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*, const OutputBlockMap*);
template void pipeline_execute<double>(PipelinePlan<double>&, const GPUSparseInput<double>&, const GPUSparseInput<double>&, const DeviceSparseInput<double>&, const DeviceSparseInput<double>&);
template GeneralSparseResult<double> pipeline_extract_result<double>(PipelinePlan<double>&);

// --- general_sparse_executor.cuh functions ---
template GeneralSparseResult<double> execute_general_sparse_pipeline<double>(const GPUSparseInput<double>&, const GPUSparseInput<double>&, const DeviceSparseInput<double>&, const DeviceSparseInput<double>&, const EinsumIndexSets&, const std::string&, const DispatchConfig&, PipelineTimingBreakdown*);

} // namespace bsptc
