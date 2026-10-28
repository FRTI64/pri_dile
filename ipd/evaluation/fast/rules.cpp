// Fast rule-opponent interpreter for Protocol v4 and self-play rollouts.
//
// The Python opponent classes are tiny scalar state machines.  Running them
// through a Python method call for every (match, round) dominates rule-heavy
// evaluation and training rollouts.  This translation unit mirrors those
// classes exactly, including the per-object CPython Mersenne Twister stream,
// so the native path is bit-identical to the CPU reference.  It is compiled on
// demand by ``ipd/evaluation/fast/native.py`` and called through ctypes.
//
// Build:
//   g++ -O3 -std=c++17 -shared -fPIC -o rules.so rules.cpp
//
// The C ABI is intentionally flat: one call per round for the whole batch.

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace {

constexpr int C = 0;
constexpr int D = 1;
constexpr int MAX_ROUNDS = 100;

inline int payoff(int me, int them) {
    if (me == C) return them == C ? 6 : 0;
    return them == C ? 10 : 1;
}

enum RuleType {
    T_AllC = 0,
    T_AllD,
    T_TFT,
    T_Grim,
    T_RandomMoves,
    T_TFT2,
    T_TFTLastD,
    T_Pavlov,
    T_ContriteTFT,
    T_WSLS2,
    T_TFTForgiveOnce,
    T_TFT2LastD,
    T_TwoTitsForTat,
    T_SuspiciousTFT,
    T_Majority,
    T_GenerousTFT,
    T_TwoStrike,
    T_ForgiveTFT,
    T_TFTNoisyForgive,
    T_GenerousZD,
    T_Grudger2,
    T_FirmButFair,
    T_Mistrust,
    T_Gradual,
    T_GrudgerK,
    T_GrimForgive,
    T_ProbabilisticGrim,
    T_ResetGrim,
    T_SpitefulGrim,
    T_DelayedPunisher,
    T_Bully,
    T_Prober,
    T_Tester,
    T_RemorsefulProber,
    T_EarlyDefector,
    T_RandomThenTFT,
    T_LateDefector,
    T_Periodic,
    T_MemoryOne,
    T_Reactive,
    T_ExtortZD,
    T_HarvestFactor,
    T_DynamicWindow,
    T_Kamikaze,
    T_ProbePhase,
    T_WindowTFT,
    T_NoisyTFT,
    T_Joss,
    T_Responsive3,
    T_Responsive,
    T_Responsive2,
    T_OmegaTFTLite,
    T_TFTProbe,
    T_AdaptiveProbe,
    T_PhaseSwitcher,
    T_AdaptiveMix,
    T_ProbeExploit,
    T_DelayedExtort,
    T_MemoryTwo,
    T_MetaMix,
    T_NoiseAdaptive,
    T_ReputationGrim,
    T_ContritionCycle,
    T_EndgameHunter,
    T_PatternBreaker,
    T_ConditionalExtort,
    T_LookaheadOpponent,
    T_MirrorAdaptive,
    T_DelayedGrim,
    T_AllCWithLastD,
    T_VotingPortfolio,
    T_COUNT
};

struct Row {
    int type_id = 0;
    int ip[16] = {0};
    double dp[16] = {0.0};
    int seq[8] = {0};
    int seq_len = 0;
    int mine[MAX_ROUNDS] = {0};
    int theirs[MAX_ROUNDS] = {0};
    int n_rounds = MAX_ROUNDS;

    // CPython random.Random state.
    uint32_t mt[624] = {0};
    int mti = 0;
    uint32_t mt_init[624] = {0};
    int mti_init = 0;

    // Generic scalar state used by the state machines.  Names are documented
    // at each rule; a rule only touches its own fields.
    int s_mode = 0;
    int s_mode2 = 0;
    int s_tests = 0;
    int s_repair = 0;
    int s_punish = 0;
    int s_coop = 0;
    int s_pending = 0;
    int s_total = 0;
    int s_last_test = -100;
    int s_angry = 0;
    int s_stole = 0;
    int s_probed = 0;
    int s_retaliated = 0;
    int s_grim = 0;
    int s_cooldown = 0;
    int s_dead = 0;
    int s_pattern = 0;
    int s_last_retaliated = 0;
    int s_missed = 0;
    int s_aux[8] = {0};
    // Per-member state for VotingPortfolio (up to 8 members).
    int p_repair[8] = {0};
    int p_probed[8] = {0};
    int p_mode[8] = {0};
    int p_tests[8] = {0};
    int p_last_test[8] = {0};
    int p_pattern[8] = {0};
    int p_pending[8] = {0};
    double s_pcoop = 0.5;
};

// ---------------------------------------------------------------------------
// CPython random.Random (MT19937) compatible stream
// ---------------------------------------------------------------------------

inline uint32_t mt_u32(Row &r) {
    if (r.mti >= 624) {
        for (int i = 0; i < 624; ++i) {
            uint32_t y = (r.mt[i] & 0x80000000u)
                       | (r.mt[(i + 1) % 624] & 0x7fffffffu);
            r.mt[i] = r.mt[(i + 397) % 624] ^ (y >> 1)
                    ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        r.mti = 0;
    }
    uint32_t y = r.mt[r.mti++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

inline double py_random(Row &r) {
    uint32_t a = mt_u32(r) >> 5;
    uint32_t b = mt_u32(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

inline int py_randbelow(Row &r, int n) {
    if (n <= 0) return 0;
    int k = 0;
    for (int x = n; x; x >>= 1) ++k;
    uint32_t value = mt_u32(r) >> (32 - k);
    while (static_cast<int>(value) >= n) {
        value = mt_u32(r) >> (32 - k);
    }
    return static_cast<int>(value);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

inline int count_d(const int *a, int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) total += (a[i] == D);
    return total;
}

inline bool any_d(const int *a, int n) { return count_d(a, n) > 0; }

inline int trailing_d(const int *a, int n) {
    int run = 0;
    for (int i = n - 1; i >= 0 && a[i] == D; --i) ++run;
    return run;
}

inline bool alternating_last(const int *a, int n, int pairs) {
    if (n < pairs + 1) return false;
    for (int i = n - pairs; i < n; ++i) {
        if (a[i] == a[i - 1]) return false;
    }
    return true;
}

inline void stats(const Row &s, int t, int &nC, int &nD, int &dC, int &dD) {
    nC = nD = dC = dD = 0;
    for (int i = 1; i < t; ++i) {
        if (s.mine[i - 1] == C) {
            ++nC;
            dC += (s.theirs[i] == D);
        } else {
            ++nD;
            dD += (s.theirs[i] == D);
        }
    }
}

inline double zd_prob(const Row &s, int key) {
    return s.dp[key];
}

// Mirror the ExtortZD constructor for compound families that store chi in a
// single parameter slot (PhaseSwitcher, DelayedExtort, ...).
inline void extort_prob(double chi, double *p) {
    double c = chi - 1.0 > 1e-9 ? chi - 1.0 : 1e-9;
    double limits[3] = {
        1.0 / (c * 5.0),
        1.0 / (chi * 9.0 + 1.0),
        1.0 / (10.0 + c * 1.0),
    };
    double lam = 0.95 * (limits[0] < limits[1]
                         ? (limits[0] < limits[2] ? limits[0] : limits[2])
                         : (limits[1] < limits[2] ? limits[1] : limits[2]));
    p[0] = 1.0 - lam * c * 5.0;
    p[1] = 1.0 - lam * (chi * 9.0 + 1.0);
    p[2] = lam * (10.0 + c * 1.0);
    p[3] = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (p[i] < 0.0) p[i] = 0.0;
        if (p[i] > 1.0) p[i] = 1.0;
    }
}

// Memory-one / extortion action: t == 0 cooperates; otherwise use the
// (mine[t-1], theirs[t-1]) entry of dp[0..3].
inline int zd_act(Row &s, int t, const int *mine, const int *theirs) {
    if (t == 0) return C;
    int key;
    if (mine[t - 1] == C) key = theirs[t - 1] == C ? 0 : 1;
    else key = theirs[t - 1] == C ? 2 : 3;
    return py_random(s) < zd_prob(s, key) ? C : D;
}

inline int zd_act_chi(Row &s, int t, const int *mine, const int *theirs,
                      double chi) {
    if (t == 0) return C;
    double p[4];
    extort_prob(chi, p);
    int key;
    if (mine[t - 1] == C) key = theirs[t - 1] == C ? 0 : 1;
    else key = theirs[t - 1] == C ? 2 : 3;
    return py_random(s) < p[key] ? C : D;
}

// HarvestFactor with configurable state slot.  Mirrors the Python class,
// including the optional last-round defection and the two-round repair.
inline int harvest_act(Row &s, int t, const int *mine, const int *theirs,
                       int start, bool repairable, double p_defect,
                       bool last_d, int &repair) {
    if (last_d && t == 99) return D;
    if (repair > 0) {
        --repair;
        return C;
    }
    if (t < start) return C;
    if (repairable && t >= 2 && theirs[t - 1] == D && theirs[t - 2] == D) {
        repair = 2;
        return C;
    }
    return py_random(s) < p_defect ? D : C;
}

// Responsive3 core, shared by the standalone family and the VotingPortfolio
// "adaptive" member.  All mutable state is passed by reference.
inline int responsive3_core(Row &s, int t, int &mode, int &tests,
                            int &last_test, int &pattern,
                            double g_unprov, double g_mutual, double g_resp,
                            int min_c, int min_d, double resp, double rate_thr,
                            bool last_d, int max_tests, int cooldown) {
    if (last_d && t == 99) return D;
    if (t == 0) return C;
    int nC, nD, dC, dD;
    stats(s, t, nC, nD, dC, dD);
    double pC = nC ? static_cast<double>(dC) / nC : 0.0;
    double pD = nD ? static_cast<double>(dD) / nD : 0.0;
    double rate = static_cast<double>(count_d(s.theirs, t)) / t;
    if (pattern) {
        if (!alternating_last(s.theirs, t, 9)) {
            pattern = 0;
        } else {
            return D;
        }
    } else if (alternating_last(s.theirs, t, 9)) {
        pattern = 1;
        return D;
    }
    if (mode == 1) {
        mode = 2;
        return C;
    }
    if (mode == 2) {
        mode = 3;
        return C;
    }
    if (mode == 3) {
        if (s.theirs[t - 1] == C) {
            mode = 0;
            return C;
        }
        mode = 4;
        last_test = t;
        return D;
    }
    if (mode == 0) {
        bool enough = nC >= min_c && nD >= min_d;
        bool responsive = enough && pC <= 0.35 && pD >= 0.65
                       && pD - pC >= resp;
        bool nonresponsive = enough && (pD - pC) < 0.35 && rate >= rate_thr;
        if (nonresponsive) {
            mode = 4;
            return D;
        }
        if (s.theirs[t - 1] == D) {
            double g;
            if (s.mine[t - 1] == C) g = responsive ? g_resp : g_unprov;
            else g = responsive ? g_resp : g_mutual;
            return py_random(s) < g ? C : D;
        }
        return C;
    }
    if (tests < max_tests && t - last_test >= cooldown && nC >= min_c
            && nD >= 2 && pC <= 0.35 && pD >= 0.65 && pD - pC >= 0.40) {
        ++tests;
        last_test = t;
        mode = 1;
        return C;
    }
    return D;
}

inline int tft_probe_core(Row &s, int t, int &probed, int &repair,
                          int probe_round, int recover, bool last_d) {
    if (last_d && t == 99) return D;
    if (repair > 0) {
        --repair;
        return C;
    }
    if (t == 0 || !any_d(s.theirs, t)) {
        if (probed) return D;
        if (t == probe_round) {
            probed = 1;
            return D;
        }
        return C;
    }
    if (probed) {
        repair = recover;
        probed = 0;
        if (repair > 0) {
            --repair;
            return C;
        }
        return C;
    }
    return s.theirs[t - 1];
}

// ---------------------------------------------------------------------------
// Rule implementations
// ---------------------------------------------------------------------------

int rule_act(Row &s, int t) {
    const int *mine = s.mine;
    const int *theirs = s.theirs;
    switch (s.type_id) {
    case T_AllC:
        return C;
    case T_AllD:
        return D;
    case T_TFT:
        return t == 0 ? C : theirs[t - 1];
    case T_Grim:
        return any_d(theirs, t) ? D : C;
    case T_RandomMoves:
        return py_random(s) < s.dp[0] ? C : D;
    case T_TFT2:
        if (t == 0) return C;
        return (t >= 2 && theirs[t - 1] == D && theirs[t - 2] == D) ? D : C;
    case T_TFTLastD:
        if (t == 99) return D;
        return t == 0 ? C : theirs[t - 1];
    case T_Pavlov: {
        if (t == 0) return C;
        int p = payoff(mine[t - 1], theirs[t - 1]);
        if (p >= 3) return mine[t - 1];
        return mine[t - 1] == D ? C : D;
    }
    case T_ContriteTFT:
        if (t == 0) return C;
        if (mine[t - 1] == D && theirs[t - 1] == D) return C;
        return theirs[t - 1];
    case T_WSLS2: {
        if (t == 0) return C;
        int p = payoff(mine[t - 1], theirs[t - 1]);
        if (mine[t - 1] == C) return p >= 6 ? C : D;
        return p >= 1 ? D : C;
    }
    case T_TFTForgiveOnce:
        if (t == 0) return C;
        if (theirs[t - 1] == D && !s.s_last_retaliated) {
            s.s_last_retaliated = 1;
            return D;
        }
        s.s_last_retaliated = 0;
        return C;
    case T_TFT2LastD:
        if (t == 99) return D;
        if (t == 0) return C;
        return (t >= 2 && theirs[t - 1] == D && theirs[t - 2] == D) ? D : C;
    case T_TwoTitsForTat:
        if (t == 0) return C;
        if (s.s_punish > 0) {
            --s.s_punish;
            return D;
        }
        if (theirs[t - 1] == D) {
            s.s_punish = 1;
            return D;
        }
        return C;
    case T_SuspiciousTFT:
        return t == 0 ? D : theirs[t - 1];
    case T_Majority: {
        if (t == 0) return C;
        int k = s.ip[0];
        int start = t - k > 0 ? t - k : 0;
        int len = t - start;
        int d = count_d(theirs + start, len);
        return (2 * d > len) ? D : C;
    }
    case T_GenerousTFT:
        if (s.ip[0] && t == 99) return D;
        if (t == 0) return C;
        if (theirs[t - 1] == C) return C;
        return py_random(s) < s.dp[0] ? C : D;
    case T_TwoStrike:
        if (s.ip[1] && t == 99) return D;
        if (t == 0) return C;
        if (s.s_repair > 0) {
            --s.s_repair;
            return C;
        }
        if (theirs[t - 1] == D && count_d(theirs, t) >= s.ip[0]) {
            if (mine[t - 1] == D) {
                s.s_repair = 1;
                return C;
            }
            return D;
        }
        return C;
    case T_ForgiveTFT:
        if (s.ip[1] && t == 99) return D;
        if (t == 0) return C;
        if (s.s_repair > 0) {
            --s.s_repair;
            return C;
        }
        if (theirs[t - 1] == C) {
            s.s_missed = 0;
            return C;
        }
        if (mine[t - 1] == C && s.s_missed < s.ip[0]) {
            ++s.s_missed;
            return C;
        }
        if (mine[t - 1] == D && theirs[t - 1] == D) {
            s.s_repair = 1;
            return C;
        }
        return D;
    case T_TFTNoisyForgive:
        if (t == 0) return C;
        if (theirs[t - 1] == C) return C;
        return py_random(s) < s.dp[0] ? C : D;
    case T_GenerousZD:
    case T_ExtortZD:
        return zd_act(s, t, mine, theirs);
    case T_Grudger2:
        return count_d(theirs, t) >= 2 ? D : C;
    case T_FirmButFair:
        if (t == 0) return C;
        return theirs[t - 1] == D ? D : C;
    case T_Mistrust:
        return count_d(theirs, t) >= 2 ? D : C;
    case T_Gradual:
        if (t == 0) return C;
        if (theirs[t - 1] == D && mine[t - 1] == C) {
            ++s.s_total;
            s.s_punish = s.s_total;
        }
        if (s.s_punish > 0) {
            --s.s_punish;
            return D;
        }
        if (s.s_coop > 0) {
            --s.s_coop;
            return C;
        }
        if (s.s_punish == 0 && s.s_total && theirs[t - 1] == C
                && mine[t - 1] == D) {
            s.s_coop = 1;
            return C;
        }
        return C;
    case T_GrudgerK:
        return count_d(theirs, t) >= s.ip[0] ? D : C;
    case T_GrimForgive:
        if (t == 0) return C;
        if (theirs[t - 1] == D) {
            return py_random(s) < s.dp[0] ? C : D;
        }
        return any_d(theirs, t) ? D : C;
    case T_ProbabilisticGrim:
        if (t == 0) return C;
        if (t == 99) return D;
        if (count_d(theirs, t) > s.ip[0]) {
            return py_random(s) < s.dp[0] ? D : C;
        }
        return C;
    case T_ResetGrim:
        if (t == 0) return C;
        if (t == 99) return D;
        if (s.s_punish > 0) {
            --s.s_punish;
            return D;
        }
        if (count_d(theirs, t) > s.ip[0]) {
            s.s_punish = s.ip[1];
            return D;
        }
        return C;
    case T_SpitefulGrim:
        if (t == 0) return C;
        if (t == 99) return D;
        return count_d(theirs, t) > s.ip[0] ? D : C;
    case T_DelayedPunisher:
        if (t == 0) return C;
        if (t == 99) return D;
        if (s.s_punish > 0) {
            --s.s_punish;
            return D;
        }
        if (theirs[t - 1] == D) {
            ++s.s_pending;
            if (s.s_pending > s.ip[0]) {
                s.s_pending = 0;
                s.s_punish = s.ip[1] - 1;
                return D;
            }
        } else {
            s.s_pending = 0;
        }
        return C;
    case T_Bully:
        if (t == 0) return C;
        if (t % 17 == 16) return D;
        return C;
    case T_Prober:
        if (t == 0 || t == 1) return C;
        if (t == 2) return D;
        {
            int upto = t < 3 ? t : 3;
            if (any_d(theirs, upto)) return theirs[t - 1];
        }
        return ((t - 3) % 4 < 2) ? D : C;
    case T_Tester:
        if (t == 0) return D;
        if (theirs[0] == D) return theirs[t - 1];
        return ((t - 1) % 4 < 2) ? D : C;
    case T_RemorsefulProber:
        if (t == 0) {
            s.s_mode = 0;
            return D;
        }
        if (s.s_mode == 0) {
            if (theirs[0] == D) {
                s.s_mode = 2;
                return C;
            }
            s.s_mode = 1;
            return D;
        }
        if (s.s_mode == 1) {
            if (theirs[t - 1] == D) {
                s.s_mode = 2;
                return C;
            }
            return D;
        }
        return theirs[t - 1];
    case T_EarlyDefector:
        return t < s.ip[0] ? D : C;
    case T_RandomThenTFT:
        if (t < s.ip[0]) return py_random(s) < s.dp[0] ? C : D;
        return theirs[t - 1];
    case T_LateDefector:
        return t >= 100 - s.ip[0] ? D : C;
    case T_Periodic:
        return s.seq[(t + s.ip[0]) % s.seq_len];
    case T_MemoryOne: {
        if (t == 0) return s.ip[0];
        int key;
        if (mine[t - 1] == C) key = theirs[t - 1] == C ? 0 : 1;
        else key = theirs[t - 1] == C ? 2 : 3;
        return py_random(s) < s.dp[key] ? C : D;
    }
    case T_Reactive:
        if (t == 0) return s.ip[0];
        return py_random(s) < (theirs[t - 1] == C ? s.dp[0] : s.dp[1])
               ? C : D;
    case T_HarvestFactor:
        return harvest_act(s, t, mine, theirs, s.ip[0], s.ip[1] != 0,
                           s.dp[0], s.ip[2] != 0, s.s_repair);
    case T_DynamicWindow: {
        if (s.ip[1] && t == 99) return D;
        if (t == 0) return C;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        int d = count_d(theirs + start, len);
        if (static_cast<double>(d) / static_cast<double>(len) >= 0.5) return D;
        if (theirs[t - 1] == D) {
            return py_random(s) < s.dp[2] ? C : D;
        }
        return py_random(s) < s.dp[0] ? D : C;
    }
    case T_Kamikaze:
        if (s.s_angry) return D;
        if (s.s_stole && theirs[t - 1] == D) {
            s.s_angry = 1;
            s.s_stole = 0;
            return D;
        }
        s.s_stole = 0;
        if (t > 0 && t % s.ip[0] == 0 && py_random(s) < s.dp[0]) {
            s.s_stole = 1;
            return D;
        }
        return C;
    case T_ProbePhase:
        if (t == 99) return D;
        if (t < s.ip[0]) return py_random(s) < s.dp[0] ? C : D;
        if (s.ip[1]) return t >= s.ip[2] ? D : C;
        return t > 0 ? theirs[t - 1] : C;
    case T_WindowTFT: {
        if (t == 0) return C;
        if (t == 99) return D;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        int d = count_d(theirs + start, len);
        if (static_cast<double>(d) / static_cast<double>(len) >= s.dp[0]) return D;
        if (theirs[t - 1] == D) {
            return py_random(s) < s.dp[1] ? C : D;
        }
        return C;
    }
    case T_NoisyTFT:
        if (t == 0) return C;
        {
            int x = theirs[t - 1];
            if (py_random(s) < s.dp[0]) x = 1 - x;
            return x;
        }
    case T_Joss:
        if (t == 0) return C;
        if (theirs[t - 1] == D) return D;
        return py_random(s) < s.dp[0] ? D : C;
    case T_Responsive: {
        if (s.ip[2] && t == 99) return D;
        if (t == 0) return C;
        int nC, nD, dC, dD;
        stats(s, t, nC, nD, dC, dD);
        double pC = nC ? static_cast<double>(dC) / nC : 0.0;
        double pD = nD ? static_cast<double>(dD) / nD : 0.0;
        double rate = static_cast<double>(count_d(theirs, t)) / t;
        if (s.s_mode == 1) {  // test
            if (theirs[t - 1] == C) {
                s.s_mode = 0;
                return C;
            }
            s.s_mode = 2;  // punish
            return D;
        }
        if (s.s_mode == 0) {  // nice
            bool enough = nD >= s.ip[1] && nC >= s.ip[0];
            if (enough && rate >= s.dp[2] && (pD - pC) < s.dp[1]) {
                s.s_mode = 2;
                return D;
            }
            if (theirs[t - 1] == D) {
                double g = t <= 2 ? 0.9 : s.dp[0];
                return py_random(s) < g ? C : D;
            }
            return C;
        }
        if (s.s_tests < s.ip[3] && nC >= s.ip[0] && nD >= 2
                && pC <= 0.30 && pD >= 0.65 && pD - pC >= 0.45) {
            ++s.s_tests;
            s.s_mode = 1;
            return C;
        }
        return D;
    }
    case T_Responsive2: {
        if (s.ip[3] && t == 99) return D;
        if (t == 0) return C;
        int nC, nD, dC, dD;
        stats(s, t, nC, nD, dC, dD);
        double pC = nC ? static_cast<double>(dC) / nC : 0.0;
        double pD = nD ? static_cast<double>(dD) / nD : 0.0;
        if (s.s_mode == 1) {
            if (theirs[t - 1] == C) {
                s.s_mode = 0;
                return C;
            }
            s.s_mode = 2;
            return D;
        }
        if (s.s_mode == 0) {
            int run = trailing_d(theirs, t);
            if (run >= s.ip[0] && mine[t - 1] == C) {
                s.s_mode = 2;
                return D;
            }
            if (nC >= 8 && static_cast<double>(count_d(theirs, t)) / t >= 0.28) {
                s.s_mode = 2;
                return D;
            }
            if (theirs[t - 1] == D) {
                if (mine[t - 1] == C) {
                    if (py_random(s) < s.dp[0]) return C;
                    s.s_mode = 2;
                    return D;
                }
                return py_random(s) < s.dp[1] ? C : D;
            }
            return C;
        }
        if (s.s_tests < s.ip[4] && nC >= s.ip[1] && nD >= 2
                && pC <= 0.35 && pD >= 0.65 && pD - pC >= 0.40) {
            ++s.s_tests;
            s.s_mode = 1;
            return C;
        }
        return D;
    }
    case T_Responsive3:
        return responsive3_core(
            s, t, s.s_mode, s.s_tests, s.s_last_test, s.s_pattern,
            s.dp[0], s.dp[1], s.dp[2], s.ip[0], s.ip[1], s.dp[3],
            s.dp[4], s.ip[2] != 0, s.ip[3], s.ip[4]);
    case T_OmegaTFTLite:
        if (t == 0) return C;
        if (s.s_dead) return D;
        if (t >= 8 && alternating_last(theirs, t, 7)) {
            s.s_dead = 1;
            return D;
        }
        if (t >= 2 && mine[t - 1] == D && theirs[t - 1] == D
                && mine[t - 2] == D && theirs[t - 2] == D) {
            return C;
        }
        return theirs[t - 1];
    case T_TFTProbe:
        return tft_probe_core(s, t, s.s_probed, s.s_repair, s.ip[0],
                              s.ip[1], s.ip[2] != 0);
    case T_AdaptiveProbe: {
        if (s.ip[2] && t == 99) return D;
        if (s.s_mode == 1) {  // repair
            --s.s_repair;
            if (s.s_repair <= 0) s.s_mode = 0;
            return C;
        }
        if (s.s_mode == 0 && t >= 2 && mine[t - 1] == D
                && theirs[t - 1] == D) {
            s.s_mode = 1;
            s.s_repair = s.ip[1] - 1;
            if (s.s_repair <= 0) s.s_mode = 0;
            return C;
        }
        if (s.s_mode == 2) {  // hunt
            if (theirs[t - 1] == C) return D;
            s.s_mode = 1;
            s.s_repair = s.ip[1];
            --s.s_repair;
            if (s.s_repair <= 0) s.s_mode = 0;
            return C;
        }
        if (s.s_mode == 3) {  // probe
            if (theirs[t - 1] == C) {
                s.s_mode = 2;
                return D;
            }
            s.s_mode = 1;
            s.s_repair = s.ip[1];
            --s.s_repair;
            if (s.s_repair <= 0) s.s_mode = 0;
            return C;
        }
        if (t == s.ip[0] && !any_d(theirs, t)) {
            s.s_mode = 3;
            return D;
        }
        if (t == 0) return C;
        return theirs[t - 1];
    }
    case T_PhaseSwitcher: {
        if (t == 99) return D;
        if (t < s.ip[1]) {
            int first = s.ip[0];
            if (first == 1) return C;
            if (first == 2) return (t > 0 && t % 6 == 0) ? D : C;
            return t > 0 ? theirs[t - 1] : C;
        }
        int second = s.ip[2];
        if (second == 0) return zd_act_chi(s, t, mine, theirs, s.dp[0]);
        if (second == 1) {
            return harvest_act(s, t, mine, theirs, s.ip[1], false, 1.0,
                               true, s.s_aux[0]);
        }
        if (second == 2) return any_d(theirs, t) ? D : C;
        if (second == 3) return D;
        if (second == 4) return t > 0 ? theirs[t - 1] : C;
        return C;
    }
    case T_AdaptiveMix: {
        if (t == 99) return D;
        if (t == 0) return C;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        double rate = static_cast<double>(count_d(theirs + start, len)) / len;
        if (rate >= s.dp[0]) {
            if (theirs[t - 1] == D) return D;
            return rate < 0.65 ? C : D;
        }
        int exploit = s.ip[1];
        if (exploit == 0) return zd_act_chi(s, t, mine, theirs, s.dp[1]);
        if (exploit == 1) {
            return harvest_act(s, t, mine, theirs, 25, false, 1.0, true,
                               s.s_aux[0]);
        }
        return D;
    }
    case T_ProbeExploit: {
        if (t == 99) return D;
        if (!s.s_probed) {
            if (t < s.ip[0] - 1) return C;
            if (t == s.ip[0] - 1) return D;
            s.s_probed = 1;
            s.s_retaliated = (t > 0 && theirs[t - 1] == D) ? 1 : 0;
        }
        if (s.s_retaliated) return t > 0 ? theirs[t - 1] : C;
        if (s.ip[1] == 0) return zd_act_chi(s, t, mine, theirs, s.dp[0]);
        if (s.ip[1] == 1) {
            return harvest_act(s, t, mine, theirs, 20, false, 1.0, true,
                               s.s_aux[0]);
        }
        return D;
    }
    case T_DelayedExtort:
        if (t == 99) return D;
        if (t < s.ip[0]) return C;
        return zd_act_chi(s, t, mine, theirs, s.dp[0]);
    case T_MemoryTwo:
        if (t == 99) return D;
        if (s.s_punish > 0) {
            --s.s_punish;
            return D;
        }
        if (t >= 2 && theirs[t - 1] == D && theirs[t - 2] == D) {
            s.s_punish = s.ip[0] - 1;
            return D;
        }
        if (t > 0 && theirs[t - 1] == D && !s.ip[1]) return D;
        return C;
    case T_MetaMix: {
        if (t == 99) return D;
        int mode = s.ip[0];
        if (mode == 0) return t > 0 ? theirs[t - 1] : C;
        if (mode == 1) return any_d(theirs, t) ? D : C;
        if (mode == 2) return zd_act_chi(s, t, mine, theirs, 3.0);
        if (mode == 3) {
            return harvest_act(s, t, mine, theirs, s.ip[1], true, 1.0,
                               true, s.s_aux[0]);
        }
        if (mode == 4) return C;
        // Python's MetaMix has no explicit "alld" branch: unknown modes,
        // including "alld", fall through to the 50/50 random action.
        return py_random(s) < 0.5 ? C : D;
    }
    case T_NoiseAdaptive: {
        if (t == 99) return D;
        if (t == 0) return C;
        if (s.s_punish > 0) {
            --s.s_punish;
            return D;
        }
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        double rate = static_cast<double>(count_d(theirs + start, len)) / len;
        if (rate >= s.dp[0]) {
            s.s_punish = s.ip[1] - 1;
            return D;
        }
        if (theirs[t - 1] == D) {
            return py_random(s) < s.dp[1] ? C : D;
        }
        return C;
    }
    case T_ReputationGrim:
        if (t == 99) return D;
        if (s.s_grim) {
            if (s.ip[0] && theirs[t - 1] == C) {
                ++s.s_cooldown;
                if (s.s_cooldown >= s.ip[0]) {
                    s.s_grim = 0;
                    s.s_cooldown = 0;
                    return C;
                }
            }
            return D;
        }
        if (t >= 6) {
            double cooperation = 1.0 - static_cast<double>(count_d(theirs, t)) / t;
            if (cooperation < s.dp[0]) {
                s.s_grim = 1;
                return D;
            }
        }
        return C;
    case T_ContritionCycle:
        if (t == 99) return D;
        if (t == 0) return C;
        if (s.s_repair > 0) {
            --s.s_repair;
            return C;
        }
        if (mine[t - 1] == D) {
            s.s_repair = s.ip[0];
            return C;
        }
        if (theirs[t - 1] == D) {
            ++s.s_pending;
            if (s.s_pending >= s.ip[1]) {
                s.s_pending = 0;
                s.s_repair = s.ip[0];
                return D;
            }
            return D;
        }
        s.s_pending = 0;
        return C;
    case T_EndgameHunter:
        if (t == 99) return D;
        if (t >= s.ip[0]) return D;
        if (t >= (s.ip[0] - s.ip[1] > 0 ? s.ip[0] - s.ip[1] : 0)
                && theirs[t - 1] == D) return D;
        return t > 0 ? theirs[t - 1] : C;
    case T_PatternBreaker: {
        if (t == 99) return D;
        if (t == 0) return C;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        if (len < window) return theirs[t - 1];
        double best_rate = 0.0;
        int best_period = 0;
        for (int period = 2; period <= 4; ++period) {
            int matches = 0;
            for (int i = period; i < len; ++i) {
                matches += (theirs[start + i] == theirs[start + i - period]);
            }
            double rate = static_cast<double>(matches) /
                          static_cast<double>(len - period > 0 ? len - period : 1);
            if (rate > best_rate) {
                best_rate = rate;
                best_period = period;
            }
        }
        if (best_period && best_rate >= s.dp[0]) {
            return 1 - theirs[start + len - best_period];
        }
        return theirs[t - 1];
    }
    case T_ConditionalExtort: {
        if (t == 99) return D;
        if (t == 0) return C;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        double cooperation = 1.0
            - static_cast<double>(count_d(theirs + start, len)) / len;
        if (cooperation >= s.dp[0]) {
            return zd_act_chi(s, t, mine, theirs, s.dp[1]);
        }
        return theirs[t - 1];
    }
    case T_LookaheadOpponent: {
        if (t == 99) return D;
        if (t == 0) return C;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        double observed = 1.0
            - static_cast<double>(count_d(theirs + start, len)) / len;
        s.s_pcoop = 0.7 * s.s_pcoop + 0.3 * observed;
        double p = s.s_pcoop;
        double p_c = p + 0.10 < 1.0 ? p + 0.10 : 1.0;
        double p_d = p - s.dp[1] > 0.0 ? p - s.dp[1] : 0.0;
        double gamma = s.dp[0];
        double retaliation = s.dp[1];
        int horizon = s.ip[1];
        // Inline recursive _future; horizon is small (2 or 3).
        auto future = [&](double q, int depth, auto &&self) -> double {
            if (depth <= 0) return 0.0;
            double qc = q + 0.10 < 1.0 ? q + 0.10 : 1.0;
            double qd = q - retaliation > 0.0 ? q - retaliation : 0.0;
            double vc = 6.0 * q + gamma * self(qc, depth - 1, self);
            double vd = 10.0 * q + 1.0 * (1.0 - q)
                      + gamma * self(qd, depth - 1, self);
            return vc > vd ? vc : vd;
        };
        double v_c = 6.0 * p + gamma * future(p_c, horizon - 1, future);
        double v_d = 10.0 * p + 1.0 * (1.0 - p)
                   + gamma * future(p_d, horizon - 1, future);
        return v_c >= v_d ? C : D;
    }
    case T_MirrorAdaptive: {
        if (t == 0) return C;
        if (t == 99) return D;
        int window = s.ip[0];
        int start = t - window > 0 ? t - window : 0;
        int len = t - start;
        double rate = static_cast<double>(count_d(theirs + start, len)) / len;
        if (rate >= 0.5) return D;
        if (theirs[t - 1] == D) {
            return py_random(s) < s.dp[0] ? C : D;
        }
        return C;
    }
    case T_DelayedGrim:
        if (t == 0) return C;
        return count_d(theirs, t) > s.ip[0] ? D : C;
    case T_AllCWithLastD:
        return t == 99 ? D : C;
    case T_VotingPortfolio: {
        if (t == 99) return D;
        int votes_c = 0;
        int votes_d = 0;
        for (int j = 0; j < s.seq_len; ++j) {
            int action = C;
            switch (s.seq[j]) {
            case 0:  // tft
                action = t > 0 ? theirs[t - 1] : C;
                break;
            case 1:  // grim
                action = any_d(theirs, t) ? D : C;
                break;
            case 2:  // extort
                action = zd_act_chi(s, t, mine, theirs, 3.0);
                break;
            case 3:  // harvest (start=30, repairable=True, p=1.0)
                action = harvest_act(s, t, mine, theirs, 30, true, 1.0,
                                     true, s.p_repair[j]);
                break;
            case 4:  // adaptive = Responsive3 with defaults
                action = responsive3_core(
                    s, t, s.p_mode[j], s.p_tests[j], s.p_last_test[j],
                    s.p_pattern[j], 0.55, 0.40, 0.95, 5, 3, 0.38, 0.18,
                    true, 2, 8);
                break;
            case 5:  // probe = TFTProbe(6, 2, True)
                action = tft_probe_core(s, t, s.p_probed[j], s.p_repair[j],
                                        6, 2, true);
                break;
            case 6:  // random
                action = py_random(s) < 0.5 ? C : D;
                break;
            case 7:  // coop
                action = C;
                break;
            default:  // alld
                action = D;
                break;
            }
            if (action == C) ++votes_c;
            else ++votes_d;
        }
        if (votes_c > votes_d) return C;
        if (votes_d > votes_c) return D;
        return s.ip[0] == 0 ? C : D;
    }
    default:
        return C;
    }
}

void reset_row(Row &r) {
    r.s_mode = 0;
    r.s_mode2 = 0;
    r.s_tests = 0;
    r.s_repair = 0;
    r.s_punish = 0;
    r.s_coop = 0;
    r.s_pending = 0;
    r.s_total = 0;
    r.s_last_test = -100;
    r.s_angry = 0;
    r.s_stole = 0;
    r.s_probed = 0;
    r.s_retaliated = 0;
    r.s_grim = 0;
    r.s_cooldown = 0;
    r.s_dead = 0;
    r.s_pattern = 0;
    r.s_last_retaliated = 0;
    r.s_missed = 0;
    for (int i = 0; i < 8; ++i) {
        r.s_aux[i] = 0;
        r.p_repair[i] = 0;
        r.p_probed[i] = 0;
        r.p_mode[i] = 0;
        r.p_tests[i] = 0;
        r.p_last_test[i] = -100;
        r.p_pattern[i] = 0;
        r.p_pending[i] = 0;
    }
    r.s_pcoop = 0.5;
    std::memcpy(r.mt, r.mt_init, sizeof(r.mt));
    r.mti = r.mti_init;
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" {

int rb_type_count() { return T_COUNT; }

int rb_version() { return 1; }

void *rb_create(const int *type_ids, const int *ip, const double *dp,
                const int *seq, const int *seq_len, const int *rounds,
                const unsigned int *mt, const int *mti, int n) {
    if (n <= 0) return nullptr;
    auto *rows = new Row[n];
    for (int i = 0; i < n; ++i) {
        Row &r = rows[i];
        r.type_id = type_ids[i];
        for (int k = 0; k < 16; ++k) {
            r.ip[k] = ip[i * 16 + k];
            r.dp[k] = dp[i * 16 + k];
        }
        for (int k = 0; k < 8; ++k) r.seq[k] = seq[i * 8 + k];
        r.seq_len = seq_len[i] > 0 ? seq_len[i] : 1;
        r.n_rounds = rounds[i] > 0 ? (rounds[i] > MAX_ROUNDS ? MAX_ROUNDS
                                                             : rounds[i])
                                   : MAX_ROUNDS;
        if (mt != nullptr) {
            std::memcpy(r.mt, mt + static_cast<size_t>(i) * 624,
                        sizeof(r.mt));
            std::memcpy(r.mt_init, r.mt, sizeof(r.mt));
            r.mti = mti[i];
            r.mti_init = mti[i];
        }
        reset_row(r);
    }
    return rows;
}

// Reset the whole batch to the captured initial state.
void rb_reset_all(void *handle, int n) {
    if (!handle) return;
    Row *rows = static_cast<Row *>(handle);
    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < MAX_ROUNDS; ++t) {
            rows[i].mine[t] = 0;
            rows[i].theirs[t] = 0;
        }
        reset_row(rows[i]);
    }
}

// Histories are appended by rb_observe.  Keep the current blue action out of
// the history: both sides act simultaneously.
void rb_act(void *handle, int n, int t, int *out) {
    if (!handle) return;
    Row *rows = static_cast<Row *>(handle);
    for (int i = 0; i < n; ++i) {
        out[i] = rule_act(rows[i], t);
    }
}

// Compute and commit one round in a single call.  This is the training
// rollout fast path: it avoids the act/observe round trip and the Python-side
// scatter for a fully native batch.
void rb_step(void *handle, int n, int t, const int *blue, int *out) {
    if (!handle) return;
    Row *rows = static_cast<Row *>(handle);
    for (int i = 0; i < n; ++i) {
        int action = rule_act(rows[i], t);
        out[i] = action;
        if (t >= 0 && t < MAX_ROUNDS) {
            rows[i].mine[t] = action;
            rows[i].theirs[t] = blue[i];
        }
    }
}

void rb_observe(void *handle, int n, int t, const int *blue,
                const int *opp) {
    if (!handle) return;
    Row *rows = static_cast<Row *>(handle);
    for (int i = 0; i < n; ++i) {
        if (t >= 0 && t < MAX_ROUNDS) {
            rows[i].mine[t] = opp[i];
            rows[i].theirs[t] = blue[i];
        }
    }
}

// Diagnostic accessor used by parity tests and benchmarks.  It exposes the
// first portfolio member's state plus the RNG index; it is not on the hot path.
void rb_debug(void *handle, int index, int t, int *out) {
    if (!handle || index < 0) return;
    Row &r = static_cast<Row *>(handle)[index];
    out[0] = r.s_mode;
    out[1] = r.s_tests;
    out[2] = r.s_last_test;
    out[3] = r.s_pattern;
    out[4] = r.s_repair;
    out[5] = r.s_probed;
    out[6] = r.mti;
    out[7] = r.s_punish;
    out[8] = r.p_mode[0];
    out[9] = r.p_tests[0];
    out[10] = r.p_last_test[0];
    out[11] = r.p_pattern[0];
    out[12] = r.p_mode[2];
    out[13] = r.p_tests[2];
    out[14] = r.p_last_test[2];
    out[15] = r.p_pattern[2];
    for (int k = 0; k < 4; ++k) {
        int idx = t - 2 + k;
        out[16 + k] = (idx >= 0 && idx < r.n_rounds) ? r.theirs[idx] : -1;
        out[20 + k] = (idx >= 0 && idx < r.n_rounds) ? r.mine[idx] : -1;
    }
}

void rb_destroy(void *handle) {
    delete[] static_cast<Row *>(handle);
}

}  // extern "C"
