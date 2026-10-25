// Always-cooperate opponent.
#pragma once

#include "opponent.hpp"

namespace ipd {
inline Opponent all_c() {
    return {"AllC", [](int, const std::vector<int>&,
                       const std::vector<int>&) { return C; }};
}
}  // namespace ipd
