// Dependency-free C++17 inference for the RL-Switch meta policy.
//
// Two deployable MLPs (RL-Wide160 160/128 and RL-Repair 64/32, both v1
// 85 features) are embedded by ../rl_switch_weights.hpp.  The public action
// contract matches the Python meta policy:
//
//   * default expert: RL-Wide160;
//   * after t >= 6, if both sides have at least two D in the first six
//     rounds and the opponent's latest move is C, use RL-Repair for the
//     rest of the match.
//
// Generate the weights with:
//   python3 -m ipd.tools.export_switch_cpp \
//     --base weights/rl_policy_weights_wide160.json \
//     --alt weights/rl_policy_weights_repair.json \
//     --out cpp/rl_switch_weights.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "../ipd_features.hpp"
#include "../rl_switch_weights.hpp"

namespace ipd {

struct RLSwitchParams {
    bool sample = true;
};

class RLSwitchPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit RLSwitchPolicy(std::uint32_t seed = 20261024u,
                            RLSwitchParams params = RLSwitchParams{})
        : rng_(seed), params_(params) {
        reset();
    }

    void reset() { switched_ = false; }

    bool switched() const { return switched_; }

    std::vector<double> features(const std::vector<int>& mine,
                                 const std::vector<int>& theirs) const {
        return ipd::extract_features(mine, theirs);
    }

    double p_defect(const std::vector<int>& mine,
                    const std::vector<int>& theirs) const {
        const bool use_alt = switched_;
        const std::vector<double> f = ipd::extract_features(mine, theirs);
        if (use_alt) {
            using namespace rl_switch_alt;
            return _p(f, MEAN, STD, N_FEATURES,
                      &W0[0][0], B0, N_HIDDEN0,
                      &W1[0][0], B1, N_HIDDEN1,
                      W2, B2, TEMPERATURE);
        }
        using namespace rl_switch_base;
        return _p(f, MEAN, STD, N_FEATURES,
                  &W0[0][0], B0, N_HIDDEN0,
                  &W1[0][0], B1, N_HIDDEN1,
                  W2, B2, TEMPERATURE);
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        if (t >= 99) return D;
        if (!switched_ && t >= 6
                && _count_d(theirs, 6) >= 2
                && _count_d(mine, 6) >= 2
                && !theirs.empty() && theirs.back() == C) {
            switched_ = true;
        }
        const bool use_alt = switched_;
        const bool safe = use_alt ? rl_switch_alt::SAFE
                                  : rl_switch_base::SAFE;
        if (safe && (t == 0 || !ipd::contains(theirs, D))) return C;
        const double p = p_defect(mine, theirs);
        if (!params_.sample) return p >= 0.5 ? D : C;
        return std::generate_canonical<double, 53>(rng_) < p ? D : C;
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        return act(day - 1, mine, theirs);
    }

private:
    static int _count_d(const std::vector<int>& seq, int n) {
        int count = 0;
        for (int i = 0; i < n && i < static_cast<int>(seq.size()); ++i) {
            count += (seq[i] == D);
        }
        return count;
    }

    static double _p(const std::vector<double>& f, const float* mean,
                     const float* stddev, int nf,
                     const float* w0, const float* b0, int h0,
                     const float* w1, const float* b1, int h1,
                     const float* w2, float b2, double temperature) {
        std::vector<double> x(nf);
        for (int i = 0; i < nf; ++i) {
            const double sd = stddev[i] != 0.0f ? stddev[i] : 1.0;
            x[i] = (f[i] - mean[i]) / sd;
        }
        std::vector<double> h0v(h0), h1v(h1);
        for (int j = 0; j < h0; ++j) {
            double s = b0[j];
            for (int i = 0; i < nf; ++i) s += w0[j * nf + i] * x[i];
            h0v[j] = std::tanh(s);
        }
        for (int j = 0; j < h1; ++j) {
            double s = b1[j];
            for (int i = 0; i < h0; ++i) s += w1[j * h0 + i] * h0v[i];
            h1v[j] = std::tanh(s);
        }
        double logit = b2;
        for (int i = 0; i < h1; ++i) logit += w2[i] * h1v[i];
        const double z = std::max(-40.0, std::min(40.0,
                                                  logit / std::max(1e-9, temperature)));
        return 1.0 / (1.0 + std::exp(-z));
    }

    std::mt19937 rng_;
    RLSwitchParams params_;
    bool switched_ = false;
};

}  // namespace ipd
