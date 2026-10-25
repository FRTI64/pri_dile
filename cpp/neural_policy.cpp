// Dependency-free C++17 inference for the GPU/NPU-trained neural IPD policy.
//
// The implementation lives in policies/neural_policy.hpp.  Weights are
// generated from neural_policy_weights.json by ipd/tools/export_cpp.py:
//
//     python3 -m ipd export --arch mlp
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o neural_demo
//     ./neural_demo
//
// For the reward-trained PPO weights:
//
//     python3 -m ipd export --weights weights/rl_policy_weights.json \
//         --out cpp/rl_policy_weights.hpp
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
//         -DNEURAL_POLICY_DEMO cpp/neural_policy.cpp -o rl_demo
#include "opponents/demo.hpp"
#include "policies/neural_policy.hpp"

#ifdef NEURAL_POLICY_DEMO
#include <cstdio>

int main() {
    ipd::NeuralPolicy policy(20261024u, true);
    std::printf("features=%zu\n", policy.features({}, {}).size());
    for (const auto& opponent : ipd::demo_opponents()) {
        std::printf("%-10s %4d\n", opponent.name.c_str(),
                    ipd::play_against(policy, opponent));
    }
}
#endif
