// Explicit instantiation of find_matching_pairs for float and double.
// Compiled as a separate TU to isolate gtl headers from CUTLASS headers
// (both define umul128 in the global namespace).

#include "general_sparse/pair_matching.hpp"

namespace bsptc {
template std::vector<BlockPairInfo> find_matching_pairs<float>(
    const GPUSparseInput<float>&, const GPUSparseInput<float>&,
    const ContractionConfig&);
template std::vector<BlockPairInfo> find_matching_pairs<double>(
    const GPUSparseInput<double>&, const GPUSparseInput<double>&,
    const ContractionConfig&);
} // namespace bsptc
