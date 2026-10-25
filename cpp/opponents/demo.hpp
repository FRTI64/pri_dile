// Small demo opponent set assembled from the per-opponent headers.
#pragma once

#include <vector>

#include "all_c.hpp"
#include "all_d.hpp"
#include "grim.hpp"
#include "late_d.hpp"
#include "opponent.hpp"
#include "random_p.hpp"
#include "tft.hpp"

namespace ipd {
inline std::vector<Opponent> demo_opponents() {
    return {
        all_c(),
        all_d(),
        tft(),
        grim(),
        late_defector(25),
        late_defector(50),
        random_p(0.2, 20261024u),
    };
}
}  // namespace ipd
