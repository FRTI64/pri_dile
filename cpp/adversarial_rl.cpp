// Constrained exploitation wrapper around the reward-trained RL policy.
//
// This is the C++ counterpart of ipd/policies/adversarial.py for
// base_arch="rl".  It uses the aggressive configuration:
//
//     gap=0.2, min_c=12, min_d=2, late_day=40, repair=true
//
// Both rules only trigger after the opponent has already defected, so they
// cannot trip a Grim-style unconditional retaliator.
//
// Build:
//     python3 -m ipd export --arch mlp --weights weights/rl_policy_weights.json \
//         --out cpp/rl_policy_weights.hpp
//     g++ -std=c++17 -O2 -DNEURAL_POLICY_USE_RL_WEIGHTS \
//         -DADVERSARIAL_RL_DEMO cpp/adversarial_rl.cpp -o adv_rl_demo
//     ./adv_rl_demo
#include "neural_policy.hpp"

class AdversarialRLPolicy {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit AdversarialRLPolicy(std::uint32_t seed = 20261024u,
                                 bool sample = true,
                                 double gap = 0.2,
                                 int min_c = 12,
                                 int min_d = 2,
                                 int late_day = 40,
                                 bool repair = true)
        : base_(seed, sample),
          gap_(gap),
          min_c_(min_c),
          min_d_(min_d),
          late_day_(late_day),
          repair_(repair) {
        reset();
    }

    void reset() {
        base_.reset();
        punish_ = false;
        first_d_ = -1;
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        if (t >= 99) return D;

        if (punish_) {
            if (repair_ && theirs.size() >= 2
                    && theirs[theirs.size() - 1] == C
                    && theirs[theirs.size() - 2] == C) {
                punish_ = false;
                return base_.act(t, mine, theirs);
            }
            return D;
        }

        if (first_d_ < 0 && ipd::contains(theirs, D)) {
            first_d_ = static_cast<int>(theirs.size()) - 1;
        }

        // Rule 1: a late first defection followed by a second defection is a
        // scheduled late harvester.
        if (late_day_ >= 0 && first_d_ >= late_day_
                && static_cast<int>(theirs.size()) >= first_d_ + 2
                && theirs[first_d_ + 1] == D) {
            punish_ = true;
            return D;
        }

        // Rule 2: enough evidence that the opponent does not respond to us.
        if (gap_ >= 0.0) {
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
            if (n_c >= min_c_ && n_d >= min_d_) {
                const double p_c = static_cast<double>(d_c) / n_c;
                const double p_d = static_cast<double>(d_d) / n_d;
                if (p_d - p_c <= gap_) {
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
    double gap_;
    int min_c_;
    int min_d_;
    int late_day_;
    bool repair_;
    bool punish_ = false;
    int first_d_ = -1;
};

#ifdef ADVERSARIAL_RL_DEMO
#include <cstdio>
#include <string>

static int demo_match(AdversarialRLPolicy& policy, const std::string& name,
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
    std::printf("%-10s C=%3d D=%3d score=%4d\n",
                name.c_str(), c, 100 - c, own);
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
    AdversarialRLPolicy policy;
    std::printf("features=%zu\n", ipd::extract_features({}, {}).size());
    demo_match(policy, "AllC", all_c);
    demo_match(policy, "AllD", all_d);
    demo_match(policy, "TFT", tft);
    demo_match(policy, "Grim", grim);
    return 0;
}
#endif
