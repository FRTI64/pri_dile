// Common opponent interface for the C++ demos and harnesses.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../ipd_features.hpp"

namespace ipd {

struct Opponent {
    std::string name;
    std::function<int(int, const std::vector<int>&,
                      const std::vector<int>&)> move;
};

// Play one 100-round match.  Every policy in cpp/policies exposes act().
template <class Policy>
int play_against(Policy& policy, const Opponent& opponent) {
    policy.reset();
    std::vector<int> mine, theirs;
    int score = 0;
    for (int t = 0; t < 100; ++t) {
        const int x = policy.act(t, mine, theirs);
        const int y = opponent.move(t, theirs, mine);
        mine.push_back(x);
        theirs.push_back(y);
        score += ipd::payoff(x, y);
    }
    return score;
}

}  // namespace ipd
