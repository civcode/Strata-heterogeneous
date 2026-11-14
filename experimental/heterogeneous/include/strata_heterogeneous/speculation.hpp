#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::heterogeneous {

// Row zero is the already-known next token, so at least one input row is
// accepted. Prediction from row i verifies input row i+1. The next bonus
// token is predictions[keep-1], never a prediction from a rejected row.
// Caller validates the finite result and commits the returned prefix on
// EVERY stage before exposing accepted draft tokens or the bonus token.
inline uint32_t greedy_accepted_prefix(const std::vector<int32_t>& inputs,
                                       const std::vector<int32_t>& predictions) {
    if (inputs.empty() || inputs.size() > 8 || inputs.size() != predictions.size())
        throw std::invalid_argument("invalid greedy speculation window");
    for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i] < 0 || predictions[i] < 0)
            throw std::invalid_argument("negative speculative token");
    uint32_t keep = 1;
    while (keep < inputs.size() && predictions[keep - 1] == inputs[keep])
        ++keep;
    return keep;
}

} // namespace strata::heterogeneous
