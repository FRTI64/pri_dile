// Always-defect opponent.
#pragma once

#include "opponent.hpp"

namespace ipd {
inline Opponent all_d() {
    return {"AllD", [](int, const std::vector<int>&,
                       const std::vector<int>&) { return D; }};
}
}  // namespace ipd
