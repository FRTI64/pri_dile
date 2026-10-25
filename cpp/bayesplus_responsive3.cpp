// Dependency-free C++17 implementation of:
//   - Responsive3: adaptive reciprocity with pC/pD, forgiveness, peace tests,
//     alternation detection and a dominant last-round defection.
//   - BayesPlus: Bayesian opponent-model bank, no proactive probing, last-round
//     defection.
//
// Actions: C=0, D=1.  day is 1-based.  Call reset() before each match.
// Compile demo:
//   g++ -std=c++17 -O2 -DRESP_BAYES_DEMO bayesplus_responsive3.cpp -o demo
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ipd {

constexpr int C = 0;
constexpr int D = 1;

inline int payoff(int me, int them) {
    if (me == C) return them == C ? 6 : 0;
    return them == C ? 10 : 1;
}

inline bool contains_defection(const std::vector<int>& moves) {
    return std::find(moves.begin(), moves.end(), D) != moves.end();
}

inline bool strictly_alternating(const std::vector<int>& moves, int run = 10) {
    if (static_cast<int>(moves.size()) < run) return false;
    const int start = static_cast<int>(moves.size()) - run;
    for (int i = start + 1; i < static_cast<int>(moves.size()); ++i)
        if (moves[i] == moves[i - 1]) return false;
    return true;
}

struct Stats {
    int n_c = 0, n_d = 0, d_c = 0, d_d = 0;
    double rate = 0.0;
};

inline Stats history_stats(const std::vector<int>& mine,
                           const std::vector<int>& theirs) {
    Stats s;
    for (std::size_t i = 1; i < theirs.size(); ++i) {
        if (mine[i - 1] == C) {
            ++s.n_c;
            s.d_c += (theirs[i] == D);
        } else {
            ++s.n_d;
            s.d_d += (theirs[i] == D);
        }
    }
    if (!theirs.empty()) {
        s.rate = static_cast<double>(
                     std::count(theirs.begin(), theirs.end(), D)) /
                 static_cast<double>(theirs.size());
    }
    return s;
}

// ---------------------------------------------------------------------------
// Responsive3
// ---------------------------------------------------------------------------
class Responsive3 {
public:
    explicit Responsive3(std::uint32_t seed = 20261024u) : rng_(seed) {
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

        const bool enough_c = st.n_c >= MIN_C;
        const bool enough_d = st.n_d >= MIN_D;
        const bool enough = enough_c && enough_d;
        const bool responsive =
            enough && p_c <= 0.35 && p_d >= 0.65 && gap >= RESP_GAP;
        const bool nonresponsive =
            enough && gap < 0.35 && st.rate >= DEFECT_RATE;

        if (mode_ == Nice) {
            if (nonresponsive) {
                mode_ = Punish;
                return D;
            }
            if (!theirs.empty() && theirs.back() == C) return C;
            if (theirs.empty()) return C;

            double g;
            if (responsive) {
                g = G_RESPONSIVE;
            } else if (!mine.empty() && mine.back() == C) {
                g = G_UNPROVOKED;
            } else {
                g = G_MUTUAL;
            }
            return random01() < g ? C : D;
        }

        // Punish / exploitation mode.
        if (tests_ < MAX_TESTS && day - last_test_ >= COOLDOWN &&
            st.n_c >= MIN_C && st.n_d >= 2 && p_c <= 0.35 &&
            p_d >= 0.65 && gap >= 0.40) {
            ++tests_;
            last_test_ = day;
            mode_ = Test1;
            return C;
        }
        return D;
    }

private:
    enum Mode { Nice, Punish, Test1, Test2, Test3 };

    static constexpr double G_UNPROVOKED = 0.55;
    static constexpr double G_MUTUAL = 0.40;
    static constexpr double G_RESPONSIVE = 0.95;
    static constexpr int MIN_C = 5;
    static constexpr int MIN_D = 3;
    static constexpr double RESP_GAP = 0.38;
    static constexpr double DEFECT_RATE = 0.18;
    static constexpr int MAX_TESTS = 2;
    static constexpr int COOLDOWN = 8;

    std::mt19937 rng_;
    Mode mode_ = Nice;
    int tests_ = 0;
    int last_test_ = -100;
    bool pattern_ = false;

    double random01() {
        return std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    }
};

// ---------------------------------------------------------------------------
// BayesPlus model helpers
// ---------------------------------------------------------------------------
namespace bayes {

using Pred = std::function<double(int, const std::vector<int>&,
                                  const std::vector<int>&)>;

inline double p_allc(int, const std::vector<int>&, const std::vector<int>&) {
    return 0.0;
}
inline double p_alld(int, const std::vector<int>&, const std::vector<int>&) {
    return 1.0;
}
inline double p_tft(int t, const std::vector<int>& m,
                    const std::vector<int>&) {
    return t == 0 ? 0.0 : static_cast<double>(m[t - 1] == D);
}
inline double p_stft(int t, const std::vector<int>& m,
                     const std::vector<int>&) {
    return t == 0 ? 1.0 : static_cast<double>(m[t - 1] == D);
}
inline double p_grim(int t, const std::vector<int>& m,
                     const std::vector<int>&) {
    return std::find(m.begin(), m.begin() + t, D) != m.begin() + t ? 1.0 : 0.0;
}
inline double p_tft2(int t, const std::vector<int>& m,
                     const std::vector<int>&) {
    if (t < 2) return 0.0;
    return (m[t - 1] == D && m[t - 2] == D) ? 1.0 : 0.0;
}
inline double p_gtft(int t, const std::vector<int>& m,
                     const std::vector<int>&, double g) {
    if (t == 0) return 0.0;
    return m[t - 1] == C ? 0.0 : 1.0 - g;
}
inline double p_joss(int t, const std::vector<int>& m,
                     const std::vector<int>&, double p) {
    if (t == 0) return 0.0;
    return m[t - 1] == D ? 1.0 : p;
}
inline double p_random(int, const std::vector<int>&,
                       const std::vector<int>&, double p) {
    return p;
}
inline double p_alternator(int t, const std::vector<int>&,
                           const std::vector<int>&) {
    return (t % 2 == 1) ? 1.0 : 0.0;
}
inline double p_pavlov(int t, const std::vector<int>& m,
                       const std::vector<int>& o) {
    if (t == 0) return 0.0;
    const int opp_payoff = payoff(o[t - 1], m[t - 1]);
    return opp_payoff >= 3 ? static_cast<double>(o[t - 1])
                           : static_cast<double>(1 - o[t - 1]);
}
inline double p_lated(int t, const std::vector<int>&, const std::vector<int>&,
                      int k) {
    return t >= 100 - k ? 1.0 : 0.0;
}
inline double p_noisy_tft(int t, const std::vector<int>& m,
                          const std::vector<int>&, double p) {
    if (t == 0) return 0.0;
    return m[t - 1] == C ? p : 1.0 - p;
}
inline double p_tft_delay(int t, const std::vector<int>& m,
                          const std::vector<int>&, int delay) {
    if (t < delay) return 0.0;
    return static_cast<double>(m[t - delay] == D);
}
inline double p_tft_lastd(int t, const std::vector<int>& m,
                          const std::vector<int>& o, int k) {
    if (t >= 100 - k) return 1.0;
    return p_tft(t, m, o);
}
inline double p_grudger(int t, const std::vector<int>& m,
                        const std::vector<int>&, int k) {
    return std::count(m.begin(), m.begin() + t, D) >= k ? 1.0 : 0.0;
}
inline double p_periodic(int t, const std::vector<int>&,
                         const std::vector<int>&,
                         const std::vector<int>& pattern) {
    return pattern[t % pattern.size()] == D ? 1.0 : 0.0;
}
inline double p_earlyd(int t, const std::vector<int>&,
                       const std::vector<int>&, int k) {
    return t < k ? 1.0 : 0.0;
}
inline double p_allc_lastd(int t, const std::vector<int>&,
                           const std::vector<int>&) {
    return t >= 99 ? 1.0 : 0.0;
}
inline double p_tester(int t, const std::vector<int>& m,
                       const std::vector<int>&) {
    if (t == 0) return 1.0;
    if (m[0] == D) return static_cast<double>(m[t - 1] == D);
    return ((t - 1) % 4 < 2) ? 1.0 : 0.0;
}
inline double p_prober(int t, const std::vector<int>& m,
                       const std::vector<int>&) {
    if (t < 3) return (t == 0 || t == 1) ? 0.0 : 1.0;
    const bool punished = std::find(m.begin(), m.begin() + 3, D) != m.begin() + 3;
    if (punished) return static_cast<double>(m[t - 1] == D);
    return ((t - 3) % 4 < 2) ? 1.0 : 0.0;
}
inline double p_remorseful(int t, const std::vector<int>& m,
                           const std::vector<int>&) {
    // RemorsefulProber: day 1 D.  If day 1 was punished by our D, it switches
    // to TFT starting with C on day 2.  Otherwise it exploits until it sees a
    // D from us, then switches to TFT.
    if (t == 0) return 1.0;
    bool tft = (m[0] == D);
    int action = C;

    // Day 2 (s=1)
    if (tft) {
        action = C;
    } else {
        action = D;
    }

    // Subsequent days.
    for (int s = 2; s <= t; ++s) {
        if (tft) {
            action = m[s - 1];
        } else {
            if (m[s - 1] == D) {
                tft = true;
                action = C;
            } else {
                action = D;
            }
        }
    }
    return action == D ? 1.0 : 0.0;
}

}  // namespace bayes

// ---------------------------------------------------------------------------
// BayesPlus
// ---------------------------------------------------------------------------
class BayesPlus {
public:
    BayesPlus() { reset(); }

    void reset() {
        models_.clear();
        // Base model bank (same priors as the Python reference).
        add("AllC", 8.0, bayes::p_allc);
        add("AllD", 4.0, bayes::p_alld);
        add("TFT", 18.0, bayes::p_tft);
        add("sTFT", 3.0, bayes::p_stft);
        add("Grim", 5.0, bayes::p_grim);
        add("TFT2", 8.0, bayes::p_tft2);
        add("GTFT.1", 8.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_gtft(t, m, o, 0.10);
        });
        add("GTFT.25", 3.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_gtft(t, m, o, 0.25);
        });
        add("Joss.05", 4.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_joss(t, m, o, 0.05);
        });
        add("Joss.15", 2.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_joss(t, m, o, 0.15);
        });
        add("Random.2", 3.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_random(t, m, o, 0.20);
        });
        add("Random.5", 3.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_random(t, m, o, 0.50);
        });
        add("Random.8", 1.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_random(t, m, o, 0.80);
        });
        add("Pavlov", 4.0, bayes::p_pavlov);
        add("Alternator", 1.0, bayes::p_alternator);
        for (int k : {5, 10, 15, 20, 25, 30, 40, 50}) {
            add("LateD" + std::to_string(k), 3.0,
                [k](int t, const auto& m, const auto& o) {
                    return bayes::p_lated(t, m, o, k);
                });
        }

        // BayesPlus extensions.
        for (int k = 1; k <= 99; ++k) {
            add("LateDx" + std::to_string(k), 0.16,
                [k](int t, const auto& m, const auto& o) {
                    return bayes::p_lated(t, m, o, k);
                });
        }
        for (int k : {5, 10, 20, 33, 50}) {
            add("GTFTx" + std::to_string(k), 1.5,
                [k](int t, const auto& m, const auto& o) {
                    return bayes::p_gtft(t, m, o, k / 100.0);
                });
        }
        for (double p : {0.02, 0.10, 0.20}) {
            add("NoisyTFTx" + std::to_string(p), 1.5,
                [p](int t, const auto& m, const auto& o) {
                    return bayes::p_noisy_tft(t, m, o, p);
                });
        }
        for (int d : {1, 2, 3}) {
            add("TFTdelay" + std::to_string(d), 1.5,
                [d](int t, const auto& m, const auto& o) {
                    return bayes::p_tft_delay(t, m, o, d);
                });
        }
        for (int k : {1, 2, 3, 5, 10}) {
            add("TFTlastD" + std::to_string(k), 1.0,
                [k](int t, const auto& m, const auto& o) {
                    return bayes::p_tft_lastd(t, m, o, k);
                });
        }
        for (int k : {1, 2, 3}) {
            add("Grudger" + std::to_string(k), 2.0,
                [k](int t, const auto& m, const auto& o) {
                    return bayes::p_grudger(t, m, o, k);
                });
        }
        add("Mistrust", 2.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_grudger(t, m, o, 2);
        });
        add("Periodic3", 1.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_periodic(t, m, o, {C, C, D});
        });
        add("Periodic4", 1.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_periodic(t, m, o, {C, D, D, C});
        });
        add("AlternatorD", 1.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_periodic(t, m, o, {D, C});
        });
        add("Tester", 2.0, bayes::p_tester);
        add("Prober", 2.0, bayes::p_prober);
        add("RemorsefulProber", 1.5, bayes::p_remorseful);
        add("AllC-lastD", 1.0, bayes::p_allc_lastd);
        add("EarlyD10", 1.0, [](int t, const auto& m, const auto& o) {
            return bayes::p_earlyd(t, m, o, 10);
        });
        add("EarlyD25", 0.5, [](int t, const auto& m, const auto& o) {
            return bayes::p_earlyd(t, m, o, 25);
        });

        logw_.assign(models_.size(), 0.0);
        for (std::size_t i = 0; i < models_.size(); ++i)
            logw_[i] = std::log(std::max(1e-9, models_[i].prior));
        updated_ = 0;
    }

    int move(int day, const std::vector<int>& mine,
             const std::vector<int>& theirs) {
        if (day >= 100) return D;
        if (day == 1) return C;
        // No proactive probing.
        if (!contains_defection(theirs)) return C;

        update_posterior(mine, theirs);
        double resp = 0.0, nonresp = 0.0, punitive = 0.0;
        for (std::size_t i = 0; i < models_.size(); ++i) {
            const std::string& n = models_[i].name;
            const double p = posterior_[i];
            if (starts_with(n, "TFT") || starts_with(n, "sTFT") ||
                starts_with(n, "GTFT") || starts_with(n, "Joss") ||
                starts_with(n, "NoisyTFT") || starts_with(n, "Pavlov")) {
                resp += p;
            } else if (starts_with(n, "Grim") || starts_with(n, "Grudger") ||
                       starts_with(n, "Mistrust")) {
                punitive += p;
            } else {
                nonresp += p;
            }
        }

        if (nonresp >= NONRESP_THRESHOLD || punitive >= PUNITIVE_THRESHOLD)
            return D;
        if (resp >= 0.50) return C;
        if (theirs.size() >= 2 && theirs.back() == D &&
            theirs[theirs.size() - 2] == D)
            return D;
        return C;
    }

private:
    struct Model {
        std::string name;
        double prior;
        bayes::Pred pred;
    };

    static constexpr double NONRESP_THRESHOLD = 0.40;
    static constexpr double PUNITIVE_THRESHOLD = 0.55;

    std::vector<Model> models_;
    std::vector<double> logw_;
    std::vector<double> posterior_;
    int updated_ = 0;

    static bool starts_with(const std::string& s, const std::string& p) {
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    }

    void add(const std::string& name, double prior, bayes::Pred fn) {
        models_.push_back({name, prior, std::move(fn)});
    }

    void update_posterior(const std::vector<int>& mine,
                          const std::vector<int>& theirs) {
        for (int t = updated_; t < static_cast<int>(theirs.size()); ++t) {
            const int y = theirs[t];
            for (std::size_t i = 0; i < models_.size(); ++i) {
                double p = models_[i].pred(t, mine, theirs);
                p = std::max(1e-6, std::min(1.0 - 1e-6, p));
                logw_[i] += std::log(y == D ? p : 1.0 - p);
            }
        }
        updated_ = static_cast<int>(theirs.size());
        const double mx = *std::max_element(logw_.begin(), logw_.end());
        posterior_.resize(models_.size());
        double z = 0.0;
        for (std::size_t i = 0; i < logw_.size(); ++i) {
            posterior_[i] = std::exp(logw_[i] - mx);
            z += posterior_[i];
        }
        for (double& x : posterior_) x /= z;
    }
};

}  // namespace ipd

#ifdef RESP_BAYES_DEMO
namespace {
struct Opponent {
    std::string name;
    std::function<int(int, const std::vector<int>&, const std::vector<int>&)> move;
};

int play(ipd::Responsive3& a, const Opponent& b) {
    a.reset();
    std::vector<int> am, bm;
    int score = 0;
    for (int day = 1; day <= 100; ++day) {
        const int x = a.move(day, am, bm);
        const int y = b.move(day, bm, am);
        am.push_back(x);
        bm.push_back(y);
        score += ipd::payoff(x, y);
    }
    return score;
}

int play(ipd::BayesPlus& a, const Opponent& b) {
    a.reset();
    std::vector<int> am, bm;
    int score = 0;
    for (int day = 1; day <= 100; ++day) {
        const int x = a.move(day, am, bm);
        const int y = b.move(day, bm, am);
        am.push_back(x);
        bm.push_back(y);
        score += ipd::payoff(x, y);
    }
    return score;
}
}  // namespace

int main() {
    std::vector<Opponent> opps;
    opps.push_back({"AllC", [](int, const auto&, const auto&) { return ipd::C; }});
    opps.push_back({"AllD", [](int, const auto&, const auto&) { return ipd::D; }});
    opps.push_back({"TFT", [](int t, const auto&, const auto& o) {
                        return t == 0 ? ipd::C
                                      : (o.empty() ? ipd::C : o.back());
                    }});
    opps.push_back({"Grim", [](int, const auto&, const auto& o) {
                        return ipd::contains_defection(o) ? ipd::D : ipd::C;
                    }});
    opps.push_back({"LateD25", [](int t, const auto&, const auto&) {
                        return t >= 75 ? ipd::D : ipd::C;
                    }});
    opps.push_back({"LateD50", [](int t, const auto&, const auto&) {
                        return t >= 50 ? ipd::D : ipd::C;
                    }});

    ipd::Responsive3 r3;
    ipd::BayesPlus bp;
    std::cout << "Responsive3\n";
    for (const auto& o : opps)
        std::cout << "  " << o.name << ": " << play(r3, o) << "\n";
    std::cout << "BayesPlus\n";
    for (const auto& o : opps)
        std::cout << "  " << o.name << ": " << play(bp, o) << "\n";
}
#endif
