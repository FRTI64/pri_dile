// Adaptive Final strategy for the 100-round iterated prisoner's dilemma.
//
// Moves are represented as 0 = cooperate, 1 = defect, matching the Python
// reference in ipd/core/machine.py.  `day` is 1-based.  Call reset() before
// every match.
//
// The implementation lives in policies/final_policy.hpp.  This file keeps the
// historical standalone demo entry point.
//
// Compile:
//     g++ -std=c++17 -O2 -DIPD_STRATEGY_DEMO cpp/ipd_strategy.cpp -o ipd_demo
#include "policies/final_policy.hpp"
#include "opponents/demo.hpp"

#ifdef IPD_STRATEGY_DEMO
#include <cstdio>

int main() {
    IPDStrategy strategy;
    for (const auto& opponent : ipd::demo_opponents()) {
        const int score = ipd::play_against(strategy, opponent);
        std::printf("%-10s %4d\n", opponent.name.c_str(), score);
    }
}
#endif
