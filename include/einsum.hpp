#pragma once

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <stdexcept>
#include <cstdint>

// Minimal einsum parser and index classification utilities.
// Example: parse \"ij,jk->ik\" -> inputs: [\"ij\", \"jk\"], output: \"ik\".
namespace bsptc {

// ============================================================
// Core data structures
// ============================================================

struct EinsumOperand {
    std::string labels;                     // e.g. "ij"
    std::vector<std::int64_t> shape;        // same length as labels
};

struct EinsumExpression {
    std::vector<EinsumOperand> inputs;      // input operands
    std::string output_labels;              // e.g. "ik"
};

struct EinsumIndexSets {
    std::vector<char> free_indices_A;       // appear in output, once
    std::vector<char> free_indices_B;       // appear in output, once
    std::vector<char> contracted_indices;   // summed over
    std::vector<char> batch_indices;        // appear in output and >1 input
};

// ============================================================
// Internal helpers
// ============================================================

inline std::vector<std::string>
split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        out.push_back(item);
    return out;
}

// ============================================================
// Parsing
// ============================================================
// Example: "ij,jk->ik"
// ============================================================

inline EinsumExpression
parse_einsum(const std::string& einsum) {
    auto arrow = einsum.find("->");
    if (arrow == std::string::npos)
        throw std::runtime_error("Einsum string must contain '->'");

    EinsumExpression expr;
    expr.output_labels = einsum.substr(arrow + 2);

    auto lhs = einsum.substr(0, arrow);
    auto inputs = split(lhs, ',');

    for (const auto& in : inputs) {
        EinsumOperand op;
        op.labels = in;
        expr.inputs.push_back(op);
    }

    return expr;
}

// ============================================================
// Validation
// ============================================================

inline void
validate_einsum(EinsumExpression& expr,
                const std::vector<std::vector<std::int64_t>>& shapes)
{
    if (expr.inputs.size() != shapes.size())
        throw std::runtime_error("Number of operands != number of shapes");

    for (std::size_t i = 0; i < expr.inputs.size(); ++i) {
        auto& op = expr.inputs[i];
        op.shape = shapes[i];

        if (op.labels.size() != op.shape.size())
            throw std::runtime_error("Label rank != shape rank");
    }
}

inline void create_einsum_index_sets(
    const EinsumExpression& expr,
    EinsumIndexSets& index_sets)
{
    std::map<char, int> label_counts;

    // Count occurrences of each label
    for (const auto& op : expr.inputs) {
        for (char c : op.labels) {
            label_counts[c]++;
        }
    }

    // Added to handle batched indices at later stages
    for (char c : expr.output_labels) {
        if(label_counts[c] > 1)
        label_counts[c]++;
    }

    // Classify labels
    for (char c : expr.inputs[0].labels) {
        if(label_counts[c] == 1) {
            index_sets.free_indices_A.push_back(c);
        }
        else if(label_counts[c] == 2) {
            index_sets.contracted_indices.push_back(c);
        }
        else if(label_counts[c] > 2) {
            index_sets.batch_indices.push_back(c);
        }
    }

    for (char c : expr.inputs[1].labels) {
        if(label_counts[c] == 1) {
            index_sets.free_indices_B.push_back(c);
        }
    }
}

} // namespace bsptc
