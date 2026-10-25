// Dependency-free C++17 inference for the GPU/NPU-trained GRU-Net IPD policy.
//
// The implementation lives in policies/gru_policy.hpp.  Weights are generated
// from gru_policy_weights.json by ipd/tools/export_cpp.py:
//
//     python3 -m ipd export --arch gru
//     g++ -std=c++17 -O2 -DGRU_POLICY_DEMO cpp/gru_policy.cpp -o gru_demo
//     ./gru_demo
#include "opponents/demo.hpp"
#include "policies/gru_policy.hpp"

#ifdef GRU_POLICY_DEMO
#include <cstdio>

int main() {
    ipd::GRUPolicy policy(20261024u, true);
    std::printf("features=%zu hidden=%d layers=%d\n",
                policy.features({}, {}).size(),
                gru_policy_weights::HIDDEN,
                gru_policy_weights::NUM_LAYERS);
    for (const auto& opponent : ipd::demo_opponents()) {
        std::printf("%-10s %4d\n", opponent.name.c_str(),
                    ipd::play_against(policy, opponent));
    }
}
#endif
