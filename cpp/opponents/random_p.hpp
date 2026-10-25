// Random opponent with a configurable defection probability.
#pragma once

#include <cstdint>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "opponent.hpp"

namespace ipd {
inline Opponent random_p(double p = 0.5, std::uint32_t seed = 20261024u) {
    auto rng = std::make_shared<std::mt19937>(seed);
    std::ostringstream name;
    name << "Random" << std::fixed << std::setprecision(2) << p;
    return {name.str(),
            [p, rng](int, const std::vector<int>&, const std::vector<int>&) {
                return std::generate_canonical<double, 53>(*rng) < p ? C : D;
            }};
}
}  // namespace ipd
