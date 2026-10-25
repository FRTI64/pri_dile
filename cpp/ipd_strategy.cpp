// Adaptive strategy for the 100-round iterated prisoner's dilemma.
//
// Moves are represented as 0 = cooperate, 1 = defect, matching the Python
// reference in ipd/core/machine.py.  `day` is 1-based.  Call reset() before every
// match.
//
// Usage:
//     IPDStrategy strategy;
//     strategy.reset();
//     int action = strategy.move(day, my_history, opponent_history);
//
// Compile the optional local demo with:
//     g++ -std=c++17 -O2 -DIPD_STRATEGY_DEMO ipd_strategy.cpp -o ipd_demo
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

class IPDStrategy {
public:
    static constexpr int C = 0;
    static constexpr int D = 1;

    explicit IPDStrategy(std::uint32_t seed = 20261024u)
        : rng_(seed) {
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

        const bool enough = n_c >= MIN_C && n_d >= MIN_D;
        const bool responsive =
            enough && p_c <= 0.35 && p_d >= 0.65 &&
            p_d - p_c >= RESP_GAP;
        const bool nonresponsive =
            enough && p_d - p_c < 0.35 && d_rate >= DEFECT_RATE;

        if (mode_ == Nice) {
            if (nonresponsive) {
                mode_ = Punish;
                return D;
            }
            if (theirs.back() == C) return C;

            // The opponent just defected.  If it has shown reciprocal
            // behaviour, forgive almost always; otherwise forgive the first
            // isolated defect with moderate probability to avoid noise
            // spirals.
            double p_forgive;
            if (mine.back() == C) {
                p_forgive = responsive ? P_RESPONSIVE : P_UNPROVOKED;
            } else {
                p_forgive = responsive ? P_RESPONSIVE : P_MUTUAL;
            }
            return random01() < p_forgive ? C : D;
        }

        // Punish/exploit.  Test for reciprocity at most twice, with a
        // cooldown, so a random or unconditional defector is not repeatedly
        // given free cooperative rounds.
        if (tests_ < PEACE_TESTS &&
            day - last_test_ >= PEACE_COOLDOWN &&
            enough && p_c <= 0.35 && p_d >= 0.65 &&
            p_d - p_c >= 0.40) {
            ++tests_;
            last_test_ = day;
            mode_ = Test1;
            return C;
        }
        return D;
    }

private:
    enum Mode { Nice, Punish, Test1, Test2, Test3 };

    static constexpr double P_UNPROVOKED = 0.25;
    static constexpr double P_MUTUAL = 0.45;
    static constexpr double P_RESPONSIVE = 0.95;
    static constexpr int MIN_C = 5;
    static constexpr int MIN_D = 3;
    static constexpr double RESP_GAP = 0.38;
    static constexpr double DEFECT_RATE = 0.18;
    static constexpr int PEACE_TESTS = 2;
    static constexpr int PEACE_COOLDOWN = 8;

    std::mt19937 rng_;
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

#ifdef IPD_STRATEGY_DEMO
#include <iostream>

static int play_against(IPDStrategy& s, int opponent_move) {
    std::vector<int> mine, theirs;
    int total = 0;
    for (int day = 1; day <= 100; ++day) {
        int x = s.move(day, mine, theirs);
        int y = opponent_move;
        mine.push_back(x);
        theirs.push_back(y);
        if (x == IPDStrategy::C && y == IPDStrategy::C) total += 6;
        else if (x == IPDStrategy::D && y == IPDStrategy::C) total += 10;
        else if (x == IPDStrategy::D && y == IPDStrategy::D) total += 1;
    }
    return total;
}

int main() {
    IPDStrategy s;
    s.reset();
    std::cout << "AlwaysC: " << play_against(s, IPDStrategy::C) << '\n';
    s.reset();
    std::cout << "AlwaysD: " << play_against(s, IPDStrategy::D) << '\n';
}
#endif
