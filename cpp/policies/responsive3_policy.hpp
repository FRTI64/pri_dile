// Responsive3: adaptive reciprocity with configurable thresholds.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "../ipd_features.hpp"
#include "strategy_utils.hpp"

namespace ipd {

struct Responsive3Params {
    double g_unprov = 0.55;
    double g_mutual = 0.40;
    double g_resp = 0.95;
    int min_c = 5;
    int min_d = 3;
    double resp = 0.38;
    double rate = 0.18;
    int max_tests = 2;
    int cooldown = 8;

    static Responsive3Params tuned() {
        Responsive3Params p;
        p.g_unprov = 0.31711381663242244;
        p.g_mutual = 0.26173130084075913;
        p.g_resp = 0.9804446329081643;
        p.min_c = 5;
        p.min_d = 2;
        p.resp = 0.46788262082576104;
        p.rate = 0.197064176521839;
        p.max_tests = 3;
        p.cooldown = 14;
        return p;
    }
};

class Responsive3 {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit Responsive3(std::uint32_t seed = 20261024u,
                         Responsive3Params params = Responsive3Params{})
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
        if (day >= 100) return D;
        if (day == 1) return C;

        const Stats st = history_stats(mine, theirs);
        const double p_c = st.n_c ? static_cast<double>(st.d_c) / st.n_c : 0.0;
        const double p_d = st.n_d ? static_cast<double>(st.d_d) / st.n_d : 0.0;
        const double gap = p_d - p_c;

        // A long exact alternation is treated as an autonomous pattern.
        if (pattern_) {
            if (strictly_alternating(theirs, 10)) return D;
            pattern_ = false;
        } else if (strictly_alternating(theirs, 10)) {
            pattern_ = true;
            return D;
        }

        if (mode_ == Test1) {
            mode_ = Test2;
            return C;
        }
        if (mode_ == Test2) {
            mode_ = Test3;
            return C;
        }
        if (mode_ == Test3) {
            if (!theirs.empty() && theirs.back() == C) {
                mode_ = Nice;
                return C;
            }
            mode_ = Punish;
            last_test_ = day;
            return D;
        }

        const bool enough_c = st.n_c >= params_.min_c;
        const bool enough_d = st.n_d >= params_.min_d;
        const bool enough = enough_c && enough_d;
        const bool responsive =
            enough && p_c <= 0.35 && p_d >= 0.65 && gap >= params_.resp;
        const bool nonresponsive =
            enough && gap < 0.35 && st.rate >= params_.rate;

        if (mode_ == Nice) {
            if (nonresponsive) {
                mode_ = Punish;
                return D;
            }
            if (!theirs.empty() && theirs.back() == C) return C;
            if (theirs.empty()) return C;

            double g;
            if (responsive) {
                g = params_.g_resp;
            } else if (!mine.empty() && mine.back() == C) {
                g = params_.g_unprov;
            } else {
                g = params_.g_mutual;
            }
            return random01() < g ? C : D;
        }

        // Punish / exploitation mode.
        if (tests_ < params_.max_tests &&
            day - last_test_ >= params_.cooldown &&
            st.n_c >= params_.min_c && st.n_d >= 2 && p_c <= 0.35 &&
            p_d >= 0.65 && gap >= 0.40) {
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
    Responsive3Params params_;
    Mode mode_ = Nice;
    int tests_ = 0;
    int last_test_ = -100;
    bool pattern_ = false;

    double random01() {
        return std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    }
};

}  // namespace ipd
