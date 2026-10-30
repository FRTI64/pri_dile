// Dependency-free C++17 inference for the RL-FullGuard meta policy.
//
// Base model: RL-V5-FT-Complex 160/128, 85 features, embedded by
// ../rl_full_guard_weights.hpp.  The state machine matches
// ipd/policies/meta.py::ProbeGuardPolicy:
//
//   * probe window: first D at zero-based 5/7, confirmed by three
//     consecutive D, then C,C,C;
//   * late window: first D at zero-based 9 or later, confirmed by three
//     consecutive D, then D,C,C;
//   * any cooperative reply releases back to the base policy;
//   * punishment continues until the opponent cooperates.
//
// Generate the weights with:
//   python3 -m ipd.tools.export_full_guard_cpp \
//     --weights weights/rl_policy_weights_v5_ft_complex.json \
//     --out cpp/rl_full_guard_weights.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "../ipd_features.hpp"
#include "../rl_full_guard_weights.hpp"

namespace ipd {

struct RLFullGuardParams {
    bool sample = true;
};

class RLFullGuardPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit RLFullGuardPolicy(std::uint32_t seed = 20261024u,
                               RLFullGuardParams params = RLFullGuardParams{})
        : rng_(seed), params_(params) {
        reset();
    }

    void reset() {
        state_ = BASE;
        idx_ = 0;
        current_.clear();
    }

    const char* state() const {
        return state_ == BASE ? "base" : (state_ == OBSERVE ? "observe" : "punish");
    }

    std::vector<double> features(const std::vector<int>& mine,
                                 const std::vector<int>& theirs) const {
        return ipd::extract_features(mine, theirs);
    }

    double p_defect(const std::vector<int>& mine,
                    const std::vector<int>& theirs) const {
        using namespace rl_full_guard_weights;
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
        if (state_ == BASE) {
            if (_triggered(mine, theirs)) {
                state_ = OBSERVE;
                idx_ = 0;
                return current_[0];
            }
            return _base_action(t, mine, theirs);
        }
        if (state_ == OBSERVE) {
            if (!theirs.empty() && theirs.back() == C) {
                state_ = BASE;
                return _base_action(t, mine, theirs);
            }
            ++idx_;
            if (idx_ < static_cast<int>(current_.size())) {
                return current_[idx_];
            }
            state_ = PUNISH;
            return D;
        }
        // Punish until the opponent cooperates again.
        if (!theirs.empty() && theirs.back() == C) {
            state_ = BASE;
            return _base_action(t, mine, theirs);
        }
        return D;
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        return act(day - 1, mine, theirs);
    }

private:
    enum State { BASE, OBSERVE, PUNISH };

    bool _triggered(const std::vector<int>& mine,
                    const std::vector<int>& theirs) {
        // Probe window: first D at zero-based 5 or 7.
        if (_streak_match(mine, theirs, 3,
                          [](int first) { return first == 5 || first == 7; })) {
            current_ = {C, C, C};
            return true;
        }
        // Late window: first D at zero-based 9 or later.
        if (_streak_match(mine, theirs, 3,
                          [](int first) { return first >= 9; })) {
            current_ = {D, C, C};
            return true;
        }
        return false;
    }

    template <typename FirstOk>
    bool _streak_match(const std::vector<int>& mine,
                       const std::vector<int>& theirs,
                       int n, FirstOk first_ok) const {
        if (static_cast<int>(theirs.size()) < n
                || static_cast<int>(mine.size()) < n) {
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (theirs[theirs.size() - n + i] != D) return false;
        }
        const int first = static_cast<int>(theirs.size()) - n;
        for (int i = 0; i < first; ++i) {
            if (theirs[i] == D) return false;
        }
        if (!first_ok(first)) return false;
        for (int i = 0; i < first; ++i) {
            if (mine[i] == D) return false;
        }
        return true;
    }

    int _base_action(int t, const std::vector<int>& mine,
                     const std::vector<int>& theirs) {
        if (t >= 99) return D;
        if (rl_full_guard_weights::SAFE
                && (t == 0 || !ipd::contains(theirs, D))) {
            return C;
        }
        const double p = p_defect(mine, theirs);
        if (!params_.sample) return p >= 0.5 ? D : C;
        return std::generate_canonical<double, 53>(rng_) < p ? D : C;
    }

    std::mt19937 rng_;
    RLFullGuardParams params_;
    State state_ = BASE;
    int idx_ = 0;
    std::vector<int> current_;
};

}  // namespace ipd
