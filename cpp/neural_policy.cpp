// Dependency-free C++17 inference for the GPU/NPU-trained neural IPD policy.
//
// The network weights are generated from neural_policy_weights.json by
// export_neural_cpp.py:
//
//     python3 export_neural_cpp.py
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_DEMO neural_policy.cpp -o neural_demo
//     ./neural_demo
//
// Usage:
//     NeuralPolicy policy(20261024u);
//     policy.reset();
//     int action = policy.move(day, my_history, opponent_history);
//
// Actions are 0 = cooperate, 1 = defect; day is 1-based.  The feature
// extractor lives in ipd_features.hpp and is shared with gru_policy.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "ipd_features.hpp"
#include "neural_policy_weights.hpp"

class NeuralPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit NeuralPolicy(std::uint32_t seed = 20261024u, bool sample = true)
        : rng_(seed), sample_(sample) {
        reset();
    }

    void reset() {
        // The policy is stateless; history is supplied by the harness.
    }

    std::vector<double> features(const std::vector<int>& mine,
                                 const std::vector<int>& theirs) const {
        return ipd::extract_features(mine, theirs);
    }

    double p_defect(const std::vector<int>& mine,
                    const std::vector<int>& theirs) const {
        using namespace neural_policy_weights;
        const std::vector<double> f = features(mine, theirs);
        double x[N_FEATURES];
        for (int i = 0; i < N_FEATURES; ++i) {
            const double sd = STD[i] != 0.0f ? STD[i] : 1.0;
            x[i] = (f[i] - MEAN[i]) / sd;
        }

        double h0[N_HIDDEN0];
        for (int j = 0; j < N_HIDDEN0; ++j) {
            double s = B0[j];
            for (int i = 0; i < N_FEATURES; ++i) s += W0[j][i] * x[i];
            h0[j] = std::tanh(s);
        }

        double h1[N_HIDDEN1];
        for (int j = 0; j < N_HIDDEN1; ++j) {
            double s = B1[j];
            for (int i = 0; i < N_HIDDEN0; ++i) s += W1[j][i] * h0[i];
            h1[j] = std::tanh(s);
        }

        double logit = B2;
        for (int i = 0; i < N_HIDDEN1; ++i) logit += W2[i] * h1[i];
        const double z = std::max(-40.0, std::min(40.0, logit /
                                                  std::max(1e-9, TEMPERATURE)));
        return 1.0 / (1.0 + std::exp(-z));
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        // Boundary and safety rules, matching neural_policy.py.
        if (t >= 99) return D;
        if (t == 0 || !ipd::contains(theirs, D)) return C;
        const double p = p_defect(mine, theirs);
        if (!sample_) return p >= 0.5 ? D : C;
        return std::generate_canonical<double, 53>(rng_) < p ? D : C;
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        return act(day - 1, mine, theirs);
    }

private:
    std::mt19937 rng_;
    bool sample_;
};

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
