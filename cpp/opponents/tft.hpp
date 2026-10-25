// Tit-for-tat opponent.
#pragma once

#include "opponent.hpp"

namespace ipd {
inline Opponent tft() {
    return {"TFT", [](int t, const std::vector<int>&,
                      const std::vector<int>& theirs) {
                return t == 0 || theirs.empty() ? C : theirs.back();
            }};
}
}  // namespace ipd
