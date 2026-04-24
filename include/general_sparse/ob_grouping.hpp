#pragma once

// Sort-based grouping of items by int64_t key.
// Header-only, no external dependencies (no gtl).

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace bsptc {

// CSR-style grouping result: items grouped by int64_t key.
struct OBGroupResult {
    int num_groups = 0;
    std::vector<int64_t> group_keys;     // [num_groups] unique key per group (sorted)
    std::vector<int>     group_offsets;   // [num_groups + 1] CSR offsets into group_items
    std::vector<int>     group_items;     // [n] item indices, grouped by key

    int group_size(int g) const { return group_offsets[g + 1] - group_offsets[g]; }
    const int* group_begin(int g) const { return group_items.data() + group_offsets[g]; }
    const int* group_end(int g) const { return group_items.data() + group_offsets[g + 1]; }
};

// Group n items by int64_t key.  keys[i] is the grouping key for item i.
// Sort-based: O(n log n), zero temp allocations (group_items is reused as sort buffer).
// Groups are returned in ascending key order.
inline OBGroupResult group_by_int64_key(const int64_t* keys, int n) {
    OBGroupResult result;
    if (n == 0) {
        result.group_offsets.push_back(0);
        return result;
    }

    // Sort item indices by key — group_items serves as both sort buffer and output
    result.group_items.resize(n);
    std::iota(result.group_items.begin(), result.group_items.end(), 0);
    std::sort(result.group_items.begin(), result.group_items.end(),
              [keys](int a, int b) { return keys[a] < keys[b]; });

    // Single scan: detect group boundaries in sorted order
    result.group_offsets.push_back(0);
    result.group_keys.push_back(keys[result.group_items[0]]);
    for (int i = 1; i < n; ++i) {
        if (keys[result.group_items[i]] != keys[result.group_items[i - 1]]) {
            result.group_offsets.push_back(i);
            result.group_keys.push_back(keys[result.group_items[i]]);
        }
    }
    result.group_offsets.push_back(n);
    result.num_groups = static_cast<int>(result.group_keys.size());

    return result;
}

} // namespace bsptc
