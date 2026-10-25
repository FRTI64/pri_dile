// Dependency-free C++17 inference for the GPU/NPU-trained neural IPD policy.
//
// The network weights are generated from neural_policy_weights.json by
// export_neural_cpp.py:
//
//     python3 export_neural_cpp.py
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO neural_policy.cpp -o neural_demo
//     ./neural_demo
//
// For the reward-trained PPO weights:
//
//     python3 export_neural_cpp.py --weights weights/rl_policy_weights.json \
//         --out cpp/rl_policy_weights.hpp
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
//         -DNEURAL_POLICY_DEMO neural_policy.cpp -o rl_demo
//
// Usage:
//     NeuralPolicy policy(20261024u);
//     policy.reset();
//     int action = policy.move(day, my_history, opponent_history);
//
// Actions are 0 = cooperate, 1 = defect; day is 1-based.  The feature
// extractor lives in ipd_features.hpp.
#include "neural_policy.hpp"

#ifdef NEURAL_POLICY_DEMO
#include <cstdio>
#include <string>

static int demo_match(NeuralPolicy& policy, const std::string& name,
                      int (*opponent)(int, const std::vector<int>&,
                                      const std::vector<int>&)) {
    policy.reset();
    std::vector<int> mine, theirs;
    int own = 0;
    for (int t = 0; t < 100; ++t) {
        const int x = policy.act(t, mine, theirs);
        const int y = opponent(t, theirs, mine);
        mine.push_back(x);
        theirs.push_back(y);
        own += ipd::payoff(x, y);
    }
    const int c = static_cast<int>(std::count(mine.begin(), mine.end(), 0));
    std::printf("%-8s C=%3d D=%3d score=%4d\n", name.c_str(), c, 100 - c, own);
    return own;
}

static int all_c(int, const std::vector<int>&, const std::vector<int>&) {
    return 0;
}
static int all_d(int, const std::vector<int>&, const std::vector<int>&) {
    return 1;
}
static int tft(int t, const std::vector<int>&, const std::vector<int>& theirs) {
    return t == 0 ? 0 : theirs.back();
}
static int grim(int, const std::vector<int>&, const std::vector<int>& theirs) {
    return std::find(theirs.begin(), theirs.end(), 1) != theirs.end() ? 1 : 0;
}

int main() {
    NeuralPolicy policy(20261024u, true);
    std::printf("features=%zu\n", policy.features({}, {}).size());
    demo_match(policy, "AllC", all_c);
    demo_match(policy, "AllD", all_d);
    demo_match(policy, "TFT", tft);
    demo_match(policy, "Grim", grim);
    return 0;
}
#endif
