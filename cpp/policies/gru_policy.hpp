// Dependency-free C++17 inference for the GRU-Net IPD policy.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "../gru_policy_weights.hpp"
#include "../ipd_features.hpp"

namespace ipd {

struct GRUParams {
    bool sample = true;
    bool safe = true;
    bool last_d = true;
};

class GRUPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit GRUPolicy(std::uint32_t seed = 20261024u,
                       GRUParams params = GRUParams{})
        : rng_(seed), params_(params) {
        reset();
    }

    // Backwards-compatible constructor: GRUPolicy(seed, sample).
    GRUPolicy(std::uint32_t seed, bool sample)
        : GRUPolicy(seed, GRUParams{sample}) {}

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

        if (params_.last_d && t >= 99) return D;
        if (params_.safe && (t == 0 || !ipd::contains(theirs, D))) return C;
        if (!params_.sample) return p >= 0.5 ? D : C;
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
    GRUParams params_;
    double hidden_[gru_policy_weights::NUM_LAYERS][gru_policy_weights::HIDDEN];
};

}  // namespace ipd
