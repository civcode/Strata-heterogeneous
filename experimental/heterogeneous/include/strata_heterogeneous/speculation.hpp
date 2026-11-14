#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "strata/core/pipeline_window.hpp"

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
    core::PipelineWindow window;
    window.T=static_cast<int>(inputs.size());
    std::copy(inputs.begin(),inputs.end(),window.tok);
    return static_cast<uint32_t>(1+core::pipeline_accepted_drafts(window,predictions.data()));
}

} // namespace strata::heterogeneous
