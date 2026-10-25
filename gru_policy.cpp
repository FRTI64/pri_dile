// Dependency-free C++17 inference for the GPU/NPU-trained GRU-Net IPD policy.
//
// Weights are generated from gru_policy_weights.json by export_gru_cpp.py:
//
//     python3 export_gru_cpp.py
//     g++ -std=c++17 -O2 -DGRU_POLICY_DEMO gru_policy.cpp -o gru_demo
//     ./gru_demo
//
// Usage:
//     GRUPolicy policy(20261024u);
//     policy.reset();
//     int action = policy.move(day, my_history, opponent_history);
//
// Actions are 0 = cooperate, 1 = defect; day is 1-based.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "gru_policy_weights.hpp"
#include "ipd_features.hpp"

static_assert(gru_policy_weights::NUM_LAYERS == 1,
              "the C++ GRU port currently supports one layer");

class GRUPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit GRUPolicy(std::uint32_t seed = 20261024u, bool sample = true)
        : rng_(seed), sample_(sample) {
        reset();
    }

    void reset() {
        for (int l = 0; l < gru_policy_weights::NUM_LAYERS; ++l) {
            for (int j = 0; j < gru_policy_weights::HIDDEN; ++j) {
                hidden_[l][j] = 0.0;
            }
        }
    }

    std::vector<double> features(const std::vector<int>& mine,
                                 const std::vector<int>& theirs) const {
        return ipd::extract_features(mine, theirs);
    }

    // Advances the recurrent state and returns P(defect).
    double p_defect(const std::vector<int>& mine,
                    const std::vector<int>& theirs) {
        using namespace gru_policy_weights;
        const std::vector<double> f = features(mine, theirs);
        double x[N_FEATURES];
        for (int i = 0; i < N_FEATURES; ++i) {
            const double sd = STD[i] != 0.0f ? STD[i] : 1.0;
            x[i] = (f[i] - MEAN[i]) / sd;
        }

        for (int l = 0; l < NUM_LAYERS; ++l) {
            double gi[3 * HIDDEN];
            double gh[3 * HIDDEN];
            for (int j = 0; j < 3 * HIDDEN; ++j) {
                double a = B_IH[l][j];
                for (int i = 0; i < N_FEATURES; ++i) a += W_IH[l][j][i] * x[i];
                double b = B_HH[l][j];
                for (int i = 0; i < HIDDEN; ++i) b += W_HH[l][j][i] * hidden_[l][i];
                gi[j] = a;
                gh[j] = b;
            }
            double new_h[HIDDEN];
            for (int j = 0; j < HIDDEN; ++j) {
                const double r = sigmoid(gi[j] + gh[j]);
                const double z = sigmoid(gi[HIDDEN + j] + gh[HIDDEN + j]);
                const double n = std::tanh(gi[2 * HIDDEN + j] +
                                           r * gh[2 * HIDDEN + j]);
                new_h[j] = (1.0 - z) * n + z * hidden_[l][j];
                hidden_[l][j] = new_h[j];
            }
            for (int j = 0; j < HIDDEN; ++j) x[j] = new_h[j];
        }

        double head[HIDDEN];
        for (int j = 0; j < HIDDEN; ++j) {
            double s = HEAD_B0[j];
            for (int i = 0; i < HIDDEN; ++i) s += HEAD_W0[j][i] * x[i];
            head[j] = std::tanh(s);
        }
        double logit = HEAD_B1;
        for (int i = 0; i < HIDDEN; ++i) logit += HEAD_W1[i] * head[i];
        const double z = std::max(-40.0, std::min(40.0, logit /
                                                  std::max(1e-9, TEMPERATURE)));
        return 1.0 / (1.0 + std::exp(-z));
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        // Always advance the GRU, even when a hard rule decides the action;
        // otherwise the hidden state would diverge from training.
        const double p = p_defect(mine, theirs);

        if (t >= 99) return D;
        if (t == 0 || !ipd::contains(theirs, D)) return C;
        if (!sample_) return p >= 0.5 ? D : C;
        return std::generate_canonical<double, 53>(rng_) < p ? D : C;
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        return act(day - 1, mine, theirs);
    }

private:
    static double sigmoid(double x) {
        if (x >= 0.0) {
            return 1.0 / (1.0 + std::exp(-std::min(x, 40.0)));
        }
        const double z = std::exp(std::max(x, -40.0));
        return z / (1.0 + z);
    }

    std::mt19937 rng_;
    bool sample_;
    double hidden_[gru_policy_weights::NUM_LAYERS][gru_policy_weights::HIDDEN];
};

#ifdef GRU_POLICY_DEMO
#include <cstdio>
#include <string>

static int demo_match(GRUPolicy& policy, const std::string& name,
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
    GRUPolicy policy(20261024u, true);
    std::printf("features=%zu hidden=%d layers=%d\n",
                policy.features({}, {}).size(),
                gru_policy_weights::HIDDEN,
                gru_policy_weights::NUM_LAYERS);
    demo_match(policy, "AllC", all_c);
    demo_match(policy, "AllD", all_d);
    demo_match(policy, "TFT", tft);
    demo_match(policy, "Grim", grim);
    return 0;
}
#endif
