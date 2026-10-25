// Constrained exploitation wrapper around the reward-trained RL policy.
//
// The implementation lives in policies/adversarial_rl_policy.hpp.  It is the
// C++ counterpart of ipd/policies/adversarial.py for base_arch="rl".
//
// Build:
//     python3 -m ipd export --arch mlp --weights weights/rl_policy_weights.json \
//         --out cpp/rl_policy_weights.hpp
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
//         -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_demo
//     ./adv_rl_demo
#include "opponents/demo.hpp"
#include "policies/adversarial_rl_policy.hpp"

#ifdef ADVERSARIAL_RL_DEMO
#include <cstdio>

int main() {
    AdversarialRLPolicy policy;
    std::printf("features=%zu\n", ipd::extract_features({}, {}).size());
    for (const auto& opponent : ipd::demo_opponents()) {
        std::printf("%-10s %4d\n", opponent.name.c_str(),
                    ipd::play_against(policy, opponent));
    }
}
#endif
