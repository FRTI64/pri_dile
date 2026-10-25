// BayesPlus: Bayesian opponent-model bank with configurable prior scales.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "../ipd_features.hpp"
#include "strategy_utils.hpp"

namespace ipd {
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
    if (tft) {
        action = C;
    } else {
        action = D;
    }
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

struct BayesPlusParams {
    double coop_scale = 1.0;
    double punitive_scale = 1.0;
    double late_scale = 1.0;
    double noise_scale = 1.0;
    double probe_scale = 1.0;
    double nonresp = 0.40;
    double punitive_thr = 0.55;
    double resp_thr = 0.50;

    static BayesPlusParams tuned() {
        BayesPlusParams p;
        p.coop_scale = 1.8369823500713045;
        p.punitive_scale = 2.4705958305576154;
        p.late_scale = 0.5216017158776368;
        p.noise_scale = 0.5817914534720606;
        p.probe_scale = 3.315461421931959;
        p.nonresp = 0.27974873408266854;
        p.punitive_thr = 0.7531347429866336;
        return p;
    }
};

class BayesPlus {
public:
    static constexpr int C = ipd::C;
    static constexpr int D = ipd::D;

    explicit BayesPlus(BayesPlusParams params = BayesPlusParams{})
        : params_(params) {
        reset();
    }

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

        if (nonresp >= params_.nonresp || punitive >= params_.punitive_thr)
            return D;
        if (resp >= params_.resp_thr) return C;
        if (theirs.size() >= 2 && theirs.back() == D &&
            theirs[theirs.size() - 2] == D)
            return D;
        return C;
    }

    int act(int t, const std::vector<int>& mine,
            const std::vector<int>& theirs) {
        return move(t + 1, mine, theirs);
    }

private:
    struct Model {
        std::string name;
        double prior;
        bayes::Pred pred;
    };

    BayesPlusParams params_;
    std::vector<Model> models_;
    std::vector<double> logw_;
    std::vector<double> posterior_;
    int updated_ = 0;

    double scaled_prior(const std::string& name, double prior) const {
        if (starts_with(name, "LateD")) return prior * params_.late_scale;
        if (starts_with(name, "Grim") || starts_with(name, "Grudger") ||
            starts_with(name, "Mistrust"))
            return prior * params_.punitive_scale;
        if (starts_with(name, "Joss") || starts_with(name, "NoisyTFT") ||
            starts_with(name, "Random"))
            return prior * params_.noise_scale;
        if (starts_with(name, "Tester") || starts_with(name, "Prober") ||
            starts_with(name, "Remorseful"))
            return prior * params_.probe_scale;
        return prior * params_.coop_scale;
    }

    void add(const std::string& name, double prior, bayes::Pred fn) {
        models_.push_back({name, scaled_prior(name, prior), std::move(fn)});
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
