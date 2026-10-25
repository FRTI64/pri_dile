// Shared helpers for the rule-based C++ policies.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "../ipd_features.hpp"

namespace ipd {

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

inline bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace ipd
