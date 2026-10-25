// Grim-trigger opponent.
#pragma once

#include "opponent.hpp"

namespace ipd {
inline Opponent grim() {
    return {"Grim", [](int, const std::vector<int>&,
                       const std::vector<int>& theirs) {
                return ipd::contains(theirs, D) ? D : C;
            }};
}
}  // namespace ipd
