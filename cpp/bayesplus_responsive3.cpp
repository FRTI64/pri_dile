// Dependency-free C++17 implementations of Responsive3 and BayesPlus.
//
// The implementations now live in separate headers:
//   policies/responsive3_policy.hpp
//   policies/bayes_plus_policy.hpp
//
// Actions: C=0, D=1.  day is 1-based.  Call reset() before each match.
// Compile demo:
//   g++ -std=c++17 -O2 -DRESP_BAYES_DEMO cpp/bayesplus_responsive3.cpp -o demo
#include "opponents/demo.hpp"
#include "policies/bayes_plus_policy.hpp"
#include "policies/responsive3_policy.hpp"

#ifdef RESP_BAYES_DEMO
#include <cstdio>

int main() {
    ipd::Responsive3 r3;
    ipd::BayesPlus bp;
    std::printf("Responsive3\n");
    for (const auto& opponent : ipd::demo_opponents()) {
        std::printf("  %-10s %4d\n", opponent.name.c_str(),
                    ipd::play_against(r3, opponent));
    }
    std::printf("BayesPlus\n");
    for (const auto& opponent : ipd::demo_opponents()) {
        std::printf("  %-10s %4d\n", opponent.name.c_str(),
                    ipd::play_against(bp, opponent));
    }
}
#endif
