// Final state-machine policy with explicit, per-instance parameters.
//
// Actions: C=0, D=1.  day is 1-based.  Call reset() before every match.
#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "../ipd_features.hpp"
#include "strategy_utils.hpp"

namespace ipd {

struct FinalParams {
    double P_UNPROVOKED = 0.25;
    double P_MUTUAL = 0.45;
    double P_RESPONSIVE = 0.95;
    int MIN_C = 5;
    int MIN_D = 3;
    double RESP_GAP = 0.38;
    double DEFECT_RATE = 0.18;
    int PEACE_TESTS = 2;
    int PEACE_COOLDOWN = 8;

    static FinalParams tuned() {
        FinalParams p;
        p.P_UNPROVOKED = 0.2395699546614404;
        p.P_MUTUAL = 0.6399155368157192;
        p.P_RESPONSIVE = 0.9544467968114831;
        p.MIN_C = 5;
        p.MIN_D = 2;
        p.RESP_GAP = 0.3871088255996472;
        p.DEFECT_RATE = 0.1170465108152342;
        p.PEACE_TESTS = 1;
        p.PEACE_COOLDOWN = 9;
        return p;
    }
};

class FinalPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit FinalPolicy(std::uint32_t seed = 20261024u,
                         FinalParams params = FinalParams{})
        : rng_(seed), params_(params) {
        reset();
    }

    void reset() {
        mode_ = Nice;
        tests_ = 0;
        last_test_ = -100;
        pattern_ = false;
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        if (day >= 100) return D;       // last round: defection is dominant
        if (day == 1) return C;

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
        const double p_c = n_c ? static_cast<double>(d_c) / n_c : 0.0;
        const double p_d = n_d ? static_cast<double>(d_d) / n_d : 0.0;
        const double d_rate =
            static_cast<double>(std::count(theirs.begin(), theirs.end(), D)) /
            static_cast<double>(theirs.size());

        // A long exact alternation is an autonomous pattern: always defect.
        if (pattern_) {
            if (alternating(theirs)) return D;
            pattern_ = false;
        } else if (alternating(theirs)) {
            pattern_ = true;
            return D;
        }

        // Two cooperative peace offers after punishment.
        if (mode_ == Test1) {
            mode_ = Test2;
            return C;
        }
        if (mode_ == Test2) {
            mode_ = Test3;
            return C;
        }
        if (mode_ == Test3) {
            if (theirs.back() == C) {
                mode_ = Nice;
                return C;
            }
            mode_ = Punish;
            last_test_ = day;
            return D;
        }

        const bool enough = n_c >= params_.MIN_C && n_d >= params_.MIN_D;
        const bool responsive =
            enough && p_c <= 0.35 && p_d >= 0.65 &&
            p_d - p_c >= params_.RESP_GAP;
        const bool nonresponsive =
            enough && p_d - p_c < 0.35 && d_rate >= params_.DEFECT_RATE;

        if (mode_ == Nice) {
            if (nonresponsive) {
                mode_ = Punish;
                return D;
            }
            if (theirs.back() == C) return C;

            double p_forgive;
            if (mine.back() == C) {
                p_forgive = responsive ? params_.P_RESPONSIVE
                                       : params_.P_UNPROVOKED;
            } else {
                p_forgive = responsive ? params_.P_RESPONSIVE
                                       : params_.P_MUTUAL;
            }
            return random01() < p_forgive ? C : D;
        }

        // Punish/exploit.  Test for reciprocity at most twice, with a
        // cooldown, so a random or unconditional defector is not repeatedly
        // given free cooperative rounds.
        if (tests_ < params_.PEACE_TESTS &&
            day - last_test_ >= params_.PEACE_COOLDOWN &&
            enough && p_c <= 0.35 && p_d >= 0.65 &&
            p_d - p_c >= 0.40) {
            ++tests_;
            last_test_ = day;
            mode_ = Test1;
            return C;
        }
        return D;
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        return move(t + 1, mine, theirs);
    }

private:
    enum Mode { Nice, Punish, Test1, Test2, Test3 };

    std::mt19937 rng_;
    FinalParams params_;
    Mode mode_ = Nice;
    int tests_ = 0;
    int last_test_ = -100;
    bool pattern_ = false;

    static bool alternating(const std::vector<int>& moves, int run = 10) {
        if (static_cast<int>(moves.size()) < run) return false;
        const int start = static_cast<int>(moves.size()) - run;
        for (int i = start + 1; i < static_cast<int>(moves.size()); ++i) {
            if (moves[i] == moves[i - 1]) return false;
        }
        return true;
    }

    double random01() {
        return std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    }
};

}  // namespace ipd

// Historical global name used by cpp/ipd_strategy.cpp and external snippets.
using IPDStrategy = ipd::FinalPolicy;
