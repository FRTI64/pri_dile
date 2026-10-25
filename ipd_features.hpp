// Shared 85-dimensional history features for the C++ neural policies.
//
// This mirrors neural_policy.extract_features exactly.  Keep the two files in
// sync if the Python feature extractor changes.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ipd {

constexpr int C = 0;
constexpr int D = 1;
constexpr int FEATURE_COUNT = 85;

inline int count(const std::vector<int>& seq, int value);
inline bool contains(const std::vector<int>& seq, int value);

inline double rate(const std::vector<int>& seq) {
    if (seq.empty()) return 0.0;
    return count(seq, D) / static_cast<double>(seq.size());
}

inline double rate_tail(const std::vector<int>& seq, int k) {
    const int n = static_cast<int>(seq.size());
    const int start = std::max(0, n - k);
    if (start >= n) return 0.0;
    int d = 0;
    for (int i = start; i < n; ++i) d += (seq[i] == D);
    return d / static_cast<double>(n - start);
}

inline int count(const std::vector<int>& seq, int value) {
    return static_cast<int>(std::count(seq.begin(), seq.end(), value));
}

inline bool contains(const std::vector<int>& seq, int value) {
    return std::find(seq.begin(), seq.end(), value) != seq.end();
}

inline int run_length(const std::vector<int>& seq) {
    if (seq.empty()) return 0;
    const int last = seq.back();
    int n = 0;
    for (auto it = seq.rbegin(); it != seq.rend(); ++it) {
        if (*it != last) break;
        ++n;
    }
    return n;
}

inline double alternation(const std::vector<int>& seq, int window) {
    const int n = static_cast<int>(seq.size());
    if (n < 2) return 0.0;
    const int start = std::max(0, n - window);
    const int length = n - start;
    if (length < 2) return 0.0;
    int changes = 0;
    for (int i = start + 1; i < n; ++i) changes += (seq[i] != seq[i - 1]);
    return changes / static_cast<double>(length - 1);
}

inline double exact_alternation(const std::vector<int>& seq, int window) {
    const int n = static_cast<int>(seq.size());
    if (n < window) return 0.0;
    for (int i = n - window + 1; i < n; ++i) {
        if (seq[i] == seq[i - 1]) return 0.0;
    }
    return 1.0;
}

inline int switches(const std::vector<int>& seq) {
    int n = 0;
    for (int i = 1; i < static_cast<int>(seq.size()); ++i) {
        n += (seq[i] != seq[i - 1]);
    }
    return n;
}

inline bool tail_all(const std::vector<int>& seq, int k, int value) {
    const int n = static_cast<int>(seq.size());
    if (n < k) return false;
    for (int i = n - k; i < n; ++i) {
        if (seq[i] != value) return false;
    }
    return true;
}

inline int last_index(const std::vector<int>& seq, int value) {
    for (int i = static_cast<int>(seq.size()) - 1; i >= 0; --i) {
        if (seq[i] == value) return i;
    }
    return -1;
}

inline int first_index(const std::vector<int>& seq, int value) {
    for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
        if (seq[i] == value) return i;
    }
    return -1;
}

inline int payoff(int me, int them) {
    if (me == C) return them == C ? 6 : 0;
    return them == C ? 10 : 1;
}

inline std::vector<double> extract_features(const std::vector<int>& mine,
                                            const std::vector<int>& theirs) {
    const int t = static_cast<int>(theirs.size());
    if (static_cast<int>(mine.size()) != t) {
        throw std::runtime_error("history lengths differ");
    }

    std::vector<double> f;
    f.reserve(FEATURE_COUNT);

    // Time / endgame.
    f.push_back(t / 99.0);
    f.push_back((99 - t) / 99.0);
    f.push_back(t == 0 ? 1.0 : 0.0);
    f.push_back(t >= 99 ? 1.0 : 0.0);
    f.push_back(t >= 95 ? 1.0 : 0.0);
    f.push_back(t >= 90 ? 1.0 : 0.0);

    // Recent actions, newest first.
    for (int i = 0; i < 8; ++i) {
        const int j = t - 1 - i;
        f.push_back(j < 0 ? 0.0 : (mine[j] == D ? 1.0 : -1.0));
    }
    for (int i = 0; i < 8; ++i) {
        const int j = t - 1 - i;
        f.push_back(j < 0 ? 0.0 : (theirs[j] == D ? 1.0 : -1.0));
    }

    // Cooperation / defection rates.
    f.push_back(rate(mine));
    f.push_back(rate(theirs));
    f.push_back(rate_tail(mine, 10));
    f.push_back(rate_tail(theirs, 3));
    f.push_back(rate_tail(theirs, 5));
    f.push_back(rate_tail(theirs, 10));
    f.push_back(rate_tail(theirs, 20));

    // Conditional response statistics.
    int n_c = 0, n_d = 0, d_c = 0, d_d = 0;
    for (int i = 1; i < t; ++i) {
        if (mine[i - 1] == C) {
            ++n_c;
            d_c += (theirs[i] == D);
        } else {
            ++n_d;
            d_d += (theirs[i] == D);
        }
    }
    const double p_c_raw = n_c ? static_cast<double>(d_c) / n_c : 0.0;
    const double p_d_raw = n_d ? static_cast<double>(d_d) / n_d : 0.0;
    const double p_c = (d_c + 1.0) / (n_c + 2.0);
    const double p_d = (d_d + 1.0) / (n_d + 2.0);
    f.push_back(p_c_raw);
    f.push_back(p_d_raw);
    f.push_back(p_c);
    f.push_back(p_d);
    f.push_back(p_d - p_c);
    f.push_back(p_d_raw - p_c_raw);
    f.push_back((n_c >= 5 && n_d >= 3) ? 1.0 : 0.0);
    f.push_back(std::min(n_c, 20) / 20.0);
    f.push_back(std::min(n_d, 20) / 20.0);
    f.push_back(std::log1p(n_c) / std::log(101.0));
    f.push_back(std::log1p(n_d) / std::log(101.0));

    // Global evidence.
    f.push_back(contains(theirs, D) ? 1.0 : 0.0);
    f.push_back(contains(mine, D) ? 1.0 : 0.0);
    f.push_back(contains(theirs, C) ? 1.0 : 0.0);
    f.push_back(count(theirs, D) / 100.0);
    f.push_back(count(mine, D) / 100.0);
    f.push_back(t / 99.0);

    // Runs and switching.
    f.push_back(std::min(run_length(theirs), 10) / 10.0);
    f.push_back(std::min(run_length(mine), 10) / 10.0);
    f.push_back(alternation(theirs, 10));
    f.push_back(alternation(theirs, 20));
    f.push_back(alternation(mine, 10));
    f.push_back(exact_alternation(theirs, 10));
    f.push_back(switches(theirs) / static_cast<double>(std::max(1, t - 1)));
    f.push_back(switches(mine) / static_cast<double>(std::max(1, t - 1)));
    f.push_back(tail_all(theirs, 5, C) ? 1.0 : 0.0);
    f.push_back(tail_all(theirs, 5, D) ? 1.0 : 0.0);

    // Last joint outcome.
    if (t) {
        const int a = mine.back(), b = theirs.back();
        f.push_back(a == C && b == C ? 1.0 : 0.0);
        f.push_back(a == C && b == D ? 1.0 : 0.0);
        f.push_back(a == D && b == C ? 1.0 : 0.0);
        f.push_back(a == D && b == D ? 1.0 : 0.0);
    } else {
        f.push_back(0.0);
        f.push_back(0.0);
        f.push_back(0.0);
        f.push_back(0.0);
    }

    // Joint-action frequencies.
    int cc = 0, cd = 0, dc = 0, dd = 0;
    for (int i = 0; i < t; ++i) {
        if (mine[i] == C) {
            if (theirs[i] == C) ++cc; else ++cd;
        } else if (theirs[i] == C) {
            ++dc;
        } else {
            ++dd;
        }
    }
    const double denom = std::max(1, t);
    f.push_back(cc / denom);
    f.push_back(cd / denom);
    f.push_back(dc / denom);
    f.push_back(dd / denom);

    int cc10 = 0, cd10 = 0, dc10 = 0, dd10 = 0;
    const int start10 = std::max(0, t - 10);
    for (int i = start10; i < t; ++i) {
        if (mine[i] == C) {
            if (theirs[i] == C) ++cc10; else ++cd10;
        } else if (theirs[i] == C) {
            ++dc10;
        } else {
            ++dd10;
        }
    }
    const double d10 = std::max(1, std::min(10, t));
    f.push_back(cc10 / d10);
    f.push_back(cd10 / d10);
    f.push_back(dc10 / d10);
    f.push_back(dd10 / d10);

    // Recency and first-defection timing.
    const int last_opp_d = last_index(theirs, D);
    const int last_my_d = last_index(mine, D);
    const int first_opp_d = first_index(theirs, D);
    const int first_my_d = first_index(mine, D);
    f.push_back(last_opp_d >= 0 ? (t - last_opp_d) / 100.0 : 1.0);
    f.push_back(last_my_d >= 0 ? (t - last_my_d) / 100.0 : 1.0);
    f.push_back(first_opp_d >= 0 ? (first_opp_d + 1) / 100.0 : 1.0);
    f.push_back(first_my_d >= 0 ? (first_my_d + 1) / 100.0 : 1.0);
    f.push_back(first_opp_d >= 0 ? (t - first_opp_d) / 100.0 : 0.0);
    f.push_back((first_my_d >= 0 && first_opp_d >= 0 && first_my_d >= first_opp_d)
                    ? (first_my_d - first_opp_d) / 100.0
                    : 0.0);

    // Score context.
    int score = 0;
    for (int i = 0; i < t; ++i) score += payoff(mine[i], theirs[i]);
    f.push_back(score / 1000.0);
    f.push_back(t ? payoff(mine.back(), theirs.back()) / 10.0 : 0.0);
    f.push_back(t ? (mine.back() == C ? 1.0 : -1.0) : 0.0);
    f.push_back(t ? (theirs.back() == C ? -1.0 : 1.0) : 0.0);
    bool last3_dd = t >= 3;
    for (int i = std::max(0, t - 3); i < t; ++i) {
        if (!(mine[i] == D && theirs[i] == D)) last3_dd = false;
    }
    f.push_back(last3_dd ? 1.0 : 0.0);

    // Response in the most recent round.
    if (t) {
        f.push_back(mine.back() == D && theirs.back() == C ? 1.0 : 0.0);
        f.push_back(mine.back() == C && theirs.back() == D ? 1.0 : 0.0);
        f.push_back(mine.back() == D && theirs.back() == D ? 1.0 : 0.0);
    } else {
        f.push_back(0.0);
        f.push_back(0.0);
        f.push_back(0.0);
    }

    // Recent conditional response (last 10 opportunities).
    int nc10 = 0, nd10 = 0, dc10r = 0, dd10r = 0;
    for (int i = std::max(1, t - 10); i < t; ++i) {
        if (mine[i - 1] == C) {
            ++nc10;
            dc10r += (theirs[i] == D);
        } else {
            ++nd10;
            dd10r += (theirs[i] == D);
        }
    }
    const double pc10 = (dc10r + 1.0) / (nc10 + 2.0);
    const double pd10 = (dd10r + 1.0) / (nd10 + 2.0);
    f.push_back(pc10);
    f.push_back(pd10);
    f.push_back(pd10 - pc10);

    if (static_cast<int>(f.size()) != FEATURE_COUNT) {
        throw std::runtime_error("feature count mismatch");
    }
    return f;
}

}  // namespace ipd
