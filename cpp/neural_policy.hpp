// Dependency-free C++17 inference for the MLP IPD policies.
//
// The same class consumes either the behavioural-cloning weights
// (neural_policy_weights.hpp) or the reward-trained PPO weights
// (rl_policy_weights.hpp / rl_policy_weights_bc.hpp).  Define
// NEURAL_POLICY_USE_RL_WEIGHTS for the final from-scratch RL weights, or
// NEURAL_POLICY_USE_RL_BC_WEIGHTS for the BC-initialised Adv-RL base.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "ipd_features.hpp"
#if defined(NEURAL_POLICY_USE_RL_BC_WEIGHTS)
#include "rl_policy_weights_bc.hpp"
#elif defined(NEURAL_POLICY_USE_RL_WEIGHTS)
#include "rl_policy_weights.hpp"
#else
#include "neural_policy_weights.hpp"
#endif

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
        if (neural_policy_weights::SAFE
                && (t == 0 || !ipd::contains(theirs, D))) return C;
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
