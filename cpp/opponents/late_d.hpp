// Late-harvesting opponent: cooperate until the last k rounds.
#pragma once

#include <string>

#include "opponent.hpp"

namespace ipd {
inline Opponent late_defector(int k = 25) {
    return {"LateD" + std::to_string(k),
            [k](int t, const std::vector<int>&, const std::vector<int>&) {
                return t >= 100 - k ? D : C;
            }};
}
}  // namespace ipd
