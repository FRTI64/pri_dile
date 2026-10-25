// Constrained exploitation wrapper around the reward-trained RL policy.
#pragma once

#include <cstdint>
#include <vector>

#include "../ipd_features.hpp"
#include "neural_policy.hpp"
#include "strategy_utils.hpp"

namespace ipd {

struct AdvRLParams {
    double gap = 0.2;
    int min_c = 12;
    int min_d = 2;
    int late_day = 40;
    bool repair = true;
};

class AdversarialRLPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit AdversarialRLPolicy(std::uint32_t seed = 20261024u,
                                 bool sample = true,
                                 AdvRLParams params = AdvRLParams{})
        : base_(seed, NeuralParams{sample}), params_(params) {
        reset();
    }

    // Historical constructor with individual tuning arguments.
    AdversarialRLPolicy(std::uint32_t seed, bool sample, double gap,
                        int min_c, int min_d, int late_day, bool repair)
        : AdversarialRLPolicy(seed, sample,
                              AdvRLParams{gap, min_c, min_d, late_day,
                                          repair}) {}

    void reset() {
        base_.reset();
        punish_ = false;
        first_d_ = -1;
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        if (t >= 99) return D;

        if (punish_) {
            if (params_.repair && theirs.size() >= 2
                    && theirs[theirs.size() - 1] == C
                    && theirs[theirs.size() - 2] == C) {
                punish_ = false;
                return base_.act(t, mine, theirs);
            }
            return D;
        }

        if (first_d_ < 0 && contains_defection(theirs)) {
            first_d_ = static_cast<int>(theirs.size()) - 1;
        }

        // Rule 1: a late first defection followed by a second defection is a
        // scheduled late harvester.
        if (params_.late_day >= 0 && first_d_ >= params_.late_day
                && static_cast<int>(theirs.size()) >= first_d_ + 2
                && theirs[first_d_ + 1] == D) {
            punish_ = true;
            return D;
        }

        // Rule 2: enough evidence that the opponent does not respond to us.
        if (params_.gap >= 0.0) {
            int n_c = 0, n_d = 0, d_c = 0, d_d = 0;
            for (std::size_t i = 1; i < theirs.size(); ++i) {
                if (mine[i - 1] == C) {
                    ++n_c;
                    d_c += (theirs[i] == D);
                } else {
                    ++n_d;
                    d_d += (theirs[i] == D);
                }
            }
            if (n_c >= params_.min_c && n_d >= params_.min_d) {
                const double p_c = static_cast<double>(d_c) / n_c;
                const double p_d = static_cast<double>(d_d) / n_d;
                if (p_d - p_c <= params_.gap) {
                    punish_ = true;
                    return D;
                }
            }
        }

        return base_.act(t, mine, theirs);
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        return act(day - 1, mine, theirs);
    }

private:
    NeuralPolicy base_;
    AdvRLParams params_;
    bool punish_ = false;
    int first_d_ = -1;
};

}  // namespace ipd

using AdversarialRLPolicy = ipd::AdversarialRLPolicy;
using AdvRLParams = ipd::AdvRLParams;
