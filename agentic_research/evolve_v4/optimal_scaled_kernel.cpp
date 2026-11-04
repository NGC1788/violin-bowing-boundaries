// Exact solvers for optimal_scaled.py (C=2 columns).  Compiled at run time by optimal_scaled.py.
//
// A joint state is code = o0 + N * o1, where o_c = sum_r d_{c,r} B^r is column c's partial
// observation (digit 0 unknown, 1+level otherwise), N = B^R, B = K + 2.  This is the same
// encoding as optimal_small.py (cell index c*R + r).  The certificate's stop test depends on a
// column only through its per-column label id (lab_c[o_c]); stopT[l0 * nl1 + l1] holds the
// frozen core.certificates() verdict "decision is not None" for a label pair.  The prior is a
// product over columns, so P(level v at (c, r) | state) = m_c[o_c + (v+1) B^r] / m_c[o_c], where
// m_c is the column's marginal likelihood of a partial observation (belief-state factorisation).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct Problem {
  int R, K, B;
  int64_t N;
  const int32_t* lab0;
  const int32_t* lab1;
  int nl1;
  const uint8_t* stopT;
  const double* m0;
  const double* m1;
  const int8_t* dig;  // N * R digits (shared by both columns)
  std::vector<int64_t> pw;
  inline bool stop(int64_t o0, int64_t o1) const {
    return stopT[(int64_t)lab0[o0] * nl1 + lab1[o1]] != 0;
  }
};

Problem make(int R, int K, int64_t N, const int32_t* lab0, const int32_t* lab1, int nl1,
             const uint8_t* stopT, const double* m0, const double* m1, const int8_t* dig) {
  Problem p{R, K, K + 2, N, lab0, lab1, nl1, stopT, m0, m1, dig, {}};
  p.pw.resize(R);
  int64_t x = 1;
  for (int r = 0; r < R; ++r) { p.pw[r] = x; x *= p.B; }
  return p;
}

const double INF = std::numeric_limits<double>::infinity();

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------------- dense DP
// Full Bellman recursion over all N*N joint codes (children have larger codes).  Same cell order
// and tie rule as optimal_small.solve (cells 0..2R-1, strict improvement by 1e-12).
// out[0] = optimal expected reads at the root, out[1] = minimax reads, out[2] = number of
// non-stopped joint states, out[3] = number of distinct states visited by the optimal policy.
int dense_solve(int R, int K, int64_t N, const int32_t* lab0, const int32_t* lab1, int nl1,
                const uint8_t* stopT, const double* m0, const double* m1, const int8_t* dig,
                double* out) {
  Problem P = make(R, K, N, lab0, lab1, nl1, stopT, m0, m1, dig);
  const int64_t NN = N * N;
  std::vector<double> V(NN);
  std::vector<uint8_t> MM(NN);
  std::vector<int8_t> act(NN, -1);
  int64_t nonstop = 0;
  for (int64_t code = NN - 1; code >= 0; --code) {
    const int64_t o0 = code % N, o1 = code / N;
    if (P.stop(o0, o1)) { V[code] = 0.0; MM[code] = 0; continue; }
    ++nonstop;
    double best = INF;
    int bcell = -1, worst = 1 << 20;
    for (int c = 0; c < 2; ++c) {
      const int64_t o = c ? o1 : o0;
      const double* m = c ? m1 : m0;
      const double inv = 1.0 / m[o];
      for (int r = 0; r < R; ++r) {
        if (dig[o * R + r]) continue;
        double e = 1.0;
        int wc = 1;
        for (int v = 0; v <= K; ++v) {
          const int64_t step = (int64_t)(v + 1) * P.pw[r];
          const int64_t ch = code + (c ? step * N : step);
          e += m[o + step] * inv * V[ch];
          wc = std::max(wc, 1 + (int)MM[ch]);
        }
        if (e < best - 1e-12) { best = e; bcell = c * R + r; }
        worst = std::min(worst, wc);
      }
    }
    V[code] = best;
    MM[code] = (uint8_t)worst;
    act[code] = (int8_t)bcell;
  }
  out[0] = V[0];
  out[1] = MM[0];
  out[2] = (double)nonstop;
  // size of the optimal policy's reachable state set
  std::vector<uint8_t> seen(NN, 0);
  std::vector<int64_t> stack{0};
  seen[0] = 1;
  int64_t tree = 0;
  while (!stack.empty()) {
    const int64_t code = stack.back();
    stack.pop_back();
    if (act[code] < 0) continue;
    ++tree;
    const int cell = act[code], c = cell / R, r = cell % R;
    for (int v = 0; v <= K; ++v) {
      const int64_t step = (int64_t)(v + 1) * P.pw[r];
      const int64_t ch = code + (c ? step * N : step);
      if (!seen[ch]) { seen[ch] = 1; stack.push_back(ch); }
    }
  }
  out[3] = (double)tree;
  return 0;
}

// ------------------------------------------------------------- single-column relaxation tables
// For the column-decomposed lower bound.  The "other" column is assumed to be fully revealed for
// free; its full label is summarised by a class f (classes = distinct stop rows).  W[o * nF + f] =
// optimal expected reads in THIS column alone until stop(label(o), f).  F[o * nF + f] = probability
// that THIS column's eventual full label falls in the other side's class f (completion distribution).
// stopRow[l * nF + f] = stop verdict of this column's label l against class f.
int column_tables(int R, int K, int64_t N, const int32_t* lab, const double* m, const int8_t* dig,
                  int nF, const uint8_t* stopRow, const int32_t* fullClass /* N, -1 if partial */,
                  int nG, double* W, float* F) {
  std::vector<int64_t> pw(R);
  int64_t x = 1;
  for (int r = 0; r < R; ++r) { pw[r] = x; x *= (K + 2); }
  std::vector<double> w(nF), f(nG);
  double* Wd = W;  // written in place (saves one N x nF temporary)
  std::vector<double> Fd((size_t)N * nG);
  for (int64_t o = N - 1; o >= 0; --o) {
    const double inv = 1.0 / m[o];
    // W
    for (int g = 0; g < nF; ++g) {
      if (stopRow[(int64_t)lab[o] * nF + g]) { w[g] = 0.0; continue; }
      double best = INF;
      for (int r = 0; r < R; ++r) {
        if (dig[o * R + r]) continue;
        double e = 1.0;
        for (int v = 0; v <= K; ++v) {
          const int64_t ch = o + (int64_t)(v + 1) * pw[r];
          e += m[ch] * inv * Wd[(size_t)ch * nF + g];
        }
        best = std::min(best, e);
      }
      w[g] = best;  // a fully observed column always stops against a full other column
    }
    for (int g = 0; g < nF; ++g) Wd[(size_t)o * nF + g] = w[g];
    // F
    int first = -1;
    for (int r = 0; r < R; ++r) if (!dig[o * R + r]) { first = r; break; }
    if (first < 0) {
      for (int g = 0; g < nG; ++g) f[g] = 0.0;
      f[fullClass[o]] = 1.0;
    } else {
      for (int g = 0; g < nG; ++g) f[g] = 0.0;
      for (int v = 0; v <= K; ++v) {
        const int64_t ch = o + (int64_t)(v + 1) * pw[first];
        const double p = m[ch] * inv;
        for (int g = 0; g < nG; ++g) f[g] += p * Fd[(size_t)ch * nG + g];
      }
    }
    for (int g = 0; g < nG; ++g) Fd[(size_t)o * nG + g] = f[g];
  }
  for (size_t i = 0; i < (size_t)N * nG; ++i) F[i] = (float)Fd[i];
  return 0;
}

// ---------------------------------------------------------------------- branch and bound
namespace {

struct Memo {
  // open addressing: key = code | exact << 63, value = exact value or lower bound
  std::vector<uint64_t> keys;
  std::vector<double> vals;
  std::vector<int8_t> acts;  // best cell of an exactly solved state (-1 otherwise)
  uint64_t mask = 0, used = 0;
  int maxlog = 28;   // never grow past 2^maxlog slots (16 bytes each); put() then refuses
  bool full = false;
  static constexpr uint64_t EMPTY = ~0ULL;
  static constexpr uint64_t EXACT = 1ULL << 63;
  void init(int logcap) {
    keys.assign(1ULL << logcap, EMPTY);
    vals.assign(1ULL << logcap, 0.0);
    acts.assign(1ULL << logcap, -1);
    mask = (1ULL << logcap) - 1;
    used = 0;
  }
  static inline uint64_t h(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    return k ^ (k >> 33);
  }
  inline int64_t find(uint64_t code) const {
    uint64_t i = h(code) & mask;
    while (keys[i] != EMPTY) {
      if ((keys[i] & ~EXACT) == code) return (int64_t)i;
      i = (i + 1) & mask;
    }
    return -1;
  }
  void grow() {
    std::vector<uint64_t> ok;
    std::vector<double> ov;
    std::vector<int8_t> oa;
    ok.swap(keys);
    ov.swap(vals);
    oa.swap(acts);
    const uint64_t cap = (mask + 1) * 2;
    keys.assign(cap, EMPTY);
    vals.assign(cap, 0.0);
    acts.assign(cap, -1);
    mask = cap - 1;
    for (size_t j = 0; j < ok.size(); ++j) {
      if (ok[j] == EMPTY) continue;
      uint64_t i = h(ok[j] & ~EXACT) & mask;
      while (keys[i] != EMPTY) i = (i + 1) & mask;
      keys[i] = ok[j];
      vals[i] = ov[j];
      acts[i] = oa[j];
    }
  }
  inline void put(uint64_t code, double v, bool exact, int act = -1) {
    if ((used + 1) * 10 > (mask + 1) * 7) {
      if ((mask + 1) >= (1ULL << maxlog)) { full = true; if ((used + 1) * 10 > (mask + 1) * 9) return; }
      else grow();
    }
    uint64_t i = h(code) & mask;
    while (keys[i] != EMPTY && (keys[i] & ~EXACT) != code) i = (i + 1) & mask;
    if (keys[i] == EMPTY) ++used;
    keys[i] = code | (exact ? EXACT : 0);
    vals[i] = v;
    acts[i] = (int8_t)act;
  }
};

struct BnB {
  Problem P;
  const double *W0, *W1;           // W0[o0*nA+a]: col-0 alone vs col-1 class a (exact, double)
  const float *F0, *F1;            // F1[o1*nA+a]: P(col-1 completion in class a | o1)
  int nA, nB;                      // nA classes of column 1 (for W0/F1), nB of column 0 (W1/F0)
  const int32_t *fz0, *fz1;        // class of a frozen column state (label can never change), else -1
  const int64_t *cn0, *cn1;        // canonical representative (per-column bisimulation) or null
  Memo memo;
  uint64_t calls = 0, expanded = 0, frozen_hits = 0;
  int64_t max_entries;
  bool overflow = false;
  double tau = 0.0;        // truncate states whose marginal probability m0*m1 < tau
  int trunc_mode = 0;      // 0: value = LB (certified lower run), 1: value = #unread (upper run)
  uint64_t truncated = 0;
  uint64_t hist[40] = {0};  // expansions by floor(-log10(m0*m1))

  Memo gmemo;  // greedy-policy values (trunc_mode 2)
  double gcut = 1e-3;  // greedy walk itself stops (value = #unread) below tau * gcut
  bool gmemo_ready = false;

  // Exact expected reads of the one-step-lookahead policy from `code` (a real policy, so an upper
  // bound on V(code)); a frozen column switches to the exact single-column optimum.
  double greedy_value(int64_t code) {
    const int64_t o0 = code % P.N, o1 = code / P.N;
    if (P.stop(o0, o1)) return 0.0;
    double fv;
    if (frozen_value(o0, o1, fv)) return fv;
    if (P.m0[o0] * P.m1[o1] < tau * gcut) {  // deep tail of the greedy walk: #unread (still an UB)
      int u = 0;
      for (int r = 0; r < P.R; ++r) u += (P.dig[o0 * P.R + r] == 0) + (P.dig[o1 * P.R + r] == 0);
      return (double)u;
    }
    if (!gmemo_ready) { gmemo.init(20); gmemo_ready = true; }
    const int64_t i = gmemo.find((uint64_t)code);
    if (i >= 0) return gmemo.vals[i];
    const int R = P.R, K = P.K;
    double bq = INF;
    int bc = -1;
    Child ch[8];
    for (int cell = 0; cell < 2 * R; ++cell) {
      const int64_t o = cell < R ? o0 : o1;
      if (P.dig[o * R + cell % R]) continue;
      if (cell < R ? fz0[o0] >= 0 : fz1[o1] >= 0) continue;
      raw_children(code, cell, ch);
      double q = 1.0;
      for (int v = 0; v <= K; ++v) {
        const int64_t a = ch[v].code % P.N, b = ch[v].code / P.N;
        double cv;
        if (P.stop(a, b)) cv = 0.0;
        else if (!frozen_value(a, b, cv)) cv = lb(a, b);
        q += ch[v].p * cv;
      }
      if (q < bq - 1e-12) { bq = q; bc = cell; }
    }
    raw_children(code, bc, ch);
    double e = 1.0;
    for (int v = 0; v <= K; ++v) e += ch[v].p * greedy_value(ch[v].code);
    gmemo.put((uint64_t)code, e, true);
    return e;
  }

  // Exact expected reads of the policy the (truncated) search chose: its action at every solved
  // state with m >= tau, the greedy policy (gcut tail = #unread) below tau.  A real policy, so an
  // upper bound on V; with trunc_mode 0 it differs from the lower bound only at truncated leaves.
  Memo pmemo;
  bool pmemo_ready = false;
  bool policy_broken = false;
  double policy_value(int64_t code) {
    const int64_t o0 = code % P.N, o1 = code / P.N;
    if (P.stop(o0, o1)) return 0.0;
    double fv;
    if (frozen_value(o0, o1, fv)) return fv;
    if (tau > 0.0 && P.m0[o0] * P.m1[o1] < tau) return greedy_value(code);
    if (!pmemo_ready) { pmemo.init(20); pmemo_ready = true; }
    const int64_t j = pmemo.find((uint64_t)code);
    if (j >= 0) return pmemo.vals[j];
    const int64_t i = memo.find(key(code));
    if (i < 0 || !(memo.keys[i] & Memo::EXACT) || memo.acts[i] < 0) { policy_broken = true; return INF; }
    Child ch[8];
    raw_children(code, memo.acts[i], ch);
    double e = 1.0;
    for (int v = 0; v <= P.K; ++v) e += ch[v].p * policy_value(ch[v].code);
    pmemo.put((uint64_t)code, e, true);
    return e;
  }

  inline bool trunc_value(int64_t o0, int64_t o1, double& v) {
    if (tau <= 0.0 || P.m0[o0] * P.m1[o1] >= tau) return false;
    if (trunc_mode == 0) { v = lb(o0, o1); return true; }
    if (trunc_mode == 2) { v = greedy_value(o0 + P.N * o1); return true; }
    int u = 0;
    for (int r = 0; r < P.R; ++r) u += (P.dig[o0 * P.R + r] == 0) + (P.dig[o1 * P.R + r] == 0);
    v = (double)u;
    return true;
  }

  inline uint64_t key(int64_t code) const {
    if (!cn0) return (uint64_t)code;
    return (uint64_t)(cn0[code % P.N] + P.N * cn1[code / P.N]);
  }
  // exact value if one column is frozen: the other column then faces a single-column problem
  inline bool frozen_value(int64_t o0, int64_t o1, double& v) const {
    if (fz1[o1] >= 0) { v = W0[(size_t)o0 * nA + fz1[o1]]; return true; }
    if (fz0[o0] >= 0) { v = W1[(size_t)o1 * nB + fz0[o0]]; return true; }
    return false;
  }

  inline double lb(int64_t o0, int64_t o1) const {
    double s = 0.0;
    const double* w0 = W0 + (size_t)o0 * nA;
    const float* f1 = F1 + (size_t)o1 * nA;
    for (int a = 0; a < nA; ++a) s += (double)f1[a] * (double)w0[a];
    const double* w1 = W1 + (size_t)o1 * nB;
    const float* f0 = F0 + (size_t)o0 * nB;
    for (int b = 0; b < nB; ++b) s += (double)f0[b] * (double)w1[b];
    s = s * (1.0 - 1e-6) - 1e-9;  // margin for float rounding of the F tables
    return std::max(1.0, s);
  }

  struct Child { int64_t code; double p; double val; bool exact; bool stopped; };
  struct Act { double qlb; int cell; };

  // value of a child for ordering / bounding: exact value, stored LB or heuristic LB
  inline void child_info(Child& ch) {
    const int64_t o0 = ch.code % P.N, o1 = ch.code / P.N;
    if (P.stop(o0, o1)) { ch.val = 0.0; ch.exact = true; ch.stopped = true; return; }
    ch.stopped = false;
    if (frozen_value(o0, o1, ch.val)) { ch.exact = true; return; }
    if (trunc_value(o0, o1, ch.val)) { ch.exact = true; return; }
    const int64_t i = memo.find(key(ch.code));
    if (i >= 0) { ch.val = memo.vals[i]; ch.exact = (memo.keys[i] & Memo::EXACT) != 0; return; }
    ch.val = lb(o0, o1);
    ch.exact = false;
  }

  void raw_children(int64_t code, int cell, Child* out) const {
    const int c = cell / P.R, r = cell % P.R;
    const int64_t o = c ? code / P.N : code % P.N;
    const double* m = c ? P.m1 : P.m0;
    const double inv = 1.0 / m[o];
    for (int v = 0; v <= P.K; ++v) {
      const int64_t step = (int64_t)(v + 1) * P.pw[r];
      out[v].code = code + (c ? step * P.N : step);
      out[v].p = m[o + step] * inv;
    }
  }

  void children(int64_t code, int cell, Child* out) {
    const int c = cell / P.R, r = cell % P.R;
    const int64_t o = c ? code / P.N : code % P.N;
    const double* m = c ? P.m1 : P.m0;
    const double inv = 1.0 / m[o];
    for (int v = 0; v <= P.K; ++v) {
      const int64_t step = (int64_t)(v + 1) * P.pw[r];
      out[v].code = code + (c ? step * P.N : step);
      out[v].p = m[o + step] * inv;
      child_info(out[v]);
    }
  }

  // Returns V(code) exactly if V(code) < beta; otherwise a lower bound >= beta.
  double solve(int64_t code, double beta) {
    ++calls;
    const int64_t o0 = code % P.N, o1 = code / P.N;
    if (P.stop(o0, o1)) return 0.0;
    {
      double fv;
      if (frozen_value(o0, o1, fv)) { ++frozen_hits; return fv; }
      if (trunc_value(o0, o1, fv)) { ++truncated; return fv; }
    }
    double lb0;
    const uint64_t kc = key(code);
    const int64_t i = memo.find(kc);
    if (i >= 0) {
      if (memo.keys[i] & Memo::EXACT) return memo.vals[i];
      lb0 = memo.vals[i];
    } else {
      lb0 = lb(o0, o1);
    }
    if (lb0 >= beta) return lb0;
    if ((int64_t)memo.used > max_entries || memo.full || gmemo.full) { overflow = true; return lb0; }
    ++expanded;
    {
      const double mm = P.m0[o0] * P.m1[o1];
      int b = mm > 0 ? (int)std::floor(-std::log10(mm)) : 39;
      hist[std::min(39, std::max(0, b))]++;
    }
    const int R = P.R, K = P.K;
    Act acts[64];
    int na = 0;
    Child tmp[8];
    for (int cell = 0; cell < 2 * R; ++cell) {
      const int c = cell / R, r = cell % R;
      const int64_t o = c ? o1 : o0;
      if (P.dig[o * R + r]) continue;
      children(code, cell, tmp);
      double q = 1.0;
      for (int v = 0; v <= K; ++v) q += tmp[v].p * tmp[v].val;
      acts[na++] = Act{q, cell};
    }
    std::stable_sort(acts, acts + na, [](const Act& a, const Act& b) { return a.qlb < b.qlb; });
    double best = beta;
    bool found = false;
    int bestcell = -1;
    double newlb = INF;
    for (int a = 0; a < na; ++a) {
      if (overflow) break;
      if (acts[a].qlb >= best) { newlb = std::min(newlb, acts[a].qlb); break; }
      Child ch[8];
      children(code, acts[a].cell, ch);
      int ord[8];
      for (int v = 0; v <= K; ++v) ord[v] = v;
      std::sort(ord, ord + K + 1, [&](int x, int y) { return ch[x].p > ch[y].p; });
      double sum = 1.0;
      for (int v = 0; v <= K; ++v) sum += ch[v].p * ch[v].val;
      bool failed = false;
      for (int j = 0; j <= K; ++j) {
        Child& c = ch[ord[j]];
        if (c.exact) continue;
        const double rest = sum - c.p * c.val;
        const double bv = (best - rest) / c.p;
        const double val = solve(c.code, bv);
        // re-read: solve stores exactness in the memo (frozen states are exact by construction)
        const int64_t k = memo.find(key(c.code));
        double dummy;
        c.exact = ((k >= 0) && (memo.keys[k] & Memo::EXACT)) ||
                  frozen_value(c.code % P.N, c.code / P.N, dummy) ||
                  trunc_value(c.code % P.N, c.code / P.N, dummy);
        c.val = val;
        sum = rest + c.p * val;
        if (overflow) { failed = true; break; }
        if (sum >= best) { failed = true; break; }
      }
      if (!failed) {
        bool allx = true;
        double s = 1.0;
        for (int v = 0; v <= K; ++v) { s += ch[v].p * ch[v].val; allx = allx && ch[v].exact; }
        if (allx && s < best) { best = s; found = true; bestcell = acts[a].cell; }
        newlb = std::min(newlb, s);
      } else {
        newlb = std::min(newlb, sum);
      }
    }
    if (overflow) return lb0;
    if (found) { memo.put(kc, best, true, bestcell); return best; }
    const double v = std::max(lb0, newlb);
    memo.put(kc, v, false);
    return v;
  }
};

}  // namespace

// out[0] = root value (exact unless out[3] != 0), out[1] = solve calls, out[2] = expanded nodes,
// out[3] = 1 if the memo hit max_entries (result is then only a lower bound), out[4] = memo size.
int bnb_solve(int R, int K, int64_t N, const int32_t* lab0, const int32_t* lab1, int nl1,
              const uint8_t* stopT, const double* m0, const double* m1, const int8_t* dig,
              const double* W0, const float* F1, int nA, const double* W1, const float* F0, int nB,
              const int32_t* fz0, const int32_t* fz1, const int64_t* cn0, const int64_t* cn1,
              double ub, int64_t max_entries, double tau, int trunc_mode, double gcut,
              double* out, double* hist) {
  BnB S{make(R, K, N, lab0, lab1, nl1, stopT, m0, m1, dig), W0, W1, F0, F1, nA, nB, fz0, fz1,
        cn0, cn1, {}, 0, 0, 0, max_entries, false};
  S.tau = tau;
  S.trunc_mode = trunc_mode;
  S.gcut = gcut;
  int ml = 22;
  while ((double)(1ULL << ml) * 0.7 < (double)max_entries && ml < 31) ++ml;
  S.memo.maxlog = ml;
  S.gmemo.maxlog = ml - 1;
  S.pmemo.maxlog = ml - 1;
  S.memo.init(22);
  const double v = S.solve(0, ub);
  out[0] = v;
  out[1] = (double)S.calls;
  out[2] = (double)S.expanded;
  out[3] = S.overflow ? 1.0 : 0.0;
  out[4] = (double)S.memo.used;
  // exactness flag of the root
  const int64_t i = S.memo.find(S.key(0));
  out[5] = (i >= 0 && (S.memo.keys[i] & Memo::EXACT)) ? 1.0 : 0.0;
  out[6] = S.lb(0, 0);
  out[7] = (double)S.frozen_hits;
  out[8] = (double)S.truncated;
  out[9] = S.gmemo_ready ? (double)S.gmemo.used : 0.0;
  out[10] = INF;
  out[11] = 0.0;
  if (out[5] > 0.5 && !S.overflow && trunc_mode == 0) {
    out[10] = S.policy_value(0);
    if (S.policy_broken || S.gmemo.full || S.pmemo.full) out[10] = INF;
    out[11] = S.pmemo_ready ? (double)S.pmemo.used : 0.0;
    out[9] = S.gmemo_ready ? (double)S.gmemo.used : 0.0;
  }
  for (int b = 0; b < 40; ++b) hist[b] = (double)S.hist[b];
  return 0;
}

// Exact expected reads of the one-step-lookahead policy "read the cell minimising
// 1 + sum_v p_v * LB(child)" (LB = the same column-decomposed bound, frozen states exact).
// Gives an upper bound on the optimum when branch and bound cannot finish.
// out[0] = expected reads, out[1] = distinct states, out[2] = 1 if max_entries was hit.
int greedy_eval(int R, int K, int64_t N, const int32_t* lab0, const int32_t* lab1, int nl1,
                const uint8_t* stopT, const double* m0, const double* m1, const int8_t* dig,
                const double* W0, const float* F1, int nA, const double* W1, const float* F0, int nB,
                const int32_t* fz0, const int32_t* fz1, int64_t max_entries, double* out) {
  BnB S{make(R, K, N, lab0, lab1, nl1, stopT, m0, m1, dig), W0, W1, F0, F1, nA, nB, fz0, fz1,
        nullptr, nullptr, {}, 0, 0, 0, max_entries, false};
  int ml = 22;
  while ((double)(1ULL << ml) * 0.7 < (double)max_entries && ml < 31) ++ml;
  S.memo.maxlog = ml;
  S.memo.init(22);
  struct Rec {
    BnB& S;
    double go(int64_t code) {
      const int64_t o0 = code % S.P.N, o1 = code / S.P.N;
      if (S.P.stop(o0, o1)) return 0.0;
      double fv;  // one column frozen: switch to the exact single-column optimum (a real policy)
      if (S.frozen_value(o0, o1, fv)) return fv;
      const int64_t i = S.memo.find((uint64_t)code);
      if (i >= 0) return S.memo.vals[i];
      if ((int64_t)S.memo.used > S.max_entries || S.memo.full) { S.overflow = true; return 0.0; }
      const int R = S.P.R, K = S.P.K;
      double bq = INF;
      int bc = -1;
      BnB::Child ch[8];
      for (int cell = 0; cell < 2 * R; ++cell) {
        const int64_t o = cell < R ? o0 : o1;
        if (S.P.dig[o * R + cell % R]) continue;
        // frozen column: its cells are useless, never pick them
        if (cell < R ? S.fz0[o0] >= 0 : S.fz1[o1] >= 0) continue;
        S.children(code, cell, ch);
        double q = 1.0;
        for (int v = 0; v <= K; ++v) {
          double cv;
          if (ch[v].stopped) cv = 0.0;
          else if (S.frozen_value(ch[v].code % S.P.N, ch[v].code / S.P.N, cv)) {}
          else cv = S.lb(ch[v].code % S.P.N, ch[v].code / S.P.N);
          q += ch[v].p * cv;
        }
        if (q < bq - 1e-12) { bq = q; bc = cell; }
      }
      S.children(code, bc, ch);
      double e = 1.0;
      for (int v = 0; v <= K; ++v) e += ch[v].p * go(ch[v].code);
      S.memo.put((uint64_t)code, e, true);
      return e;
    }
  } rec{S};
  out[0] = rec.go(0);
  out[1] = (double)S.memo.used;
  out[2] = S.overflow ? 1.0 : 0.0;
  return 0;
}

// --------------------------------------------------------------------------- hindsight
// g-tables: for every FULL column code (all digits > 0) list the labels reachable by revealing
// a subset of its cells, with the fewest cells needed.  Then hindsight(F0, F1) =
// min over (l0, g0) in list(F0) of g0 + s1[F1][l0], s1[F1][l0] = min g1 over list(F1) entries
// with stopT[l0][l1].  Expectation under the product prior (pF0, pF1).
int hindsight_exact(int R, int K, const int32_t* lab0, const int32_t* lab1, int nl0, int nl1,
                    const uint8_t* stopT, int nF0, const int64_t* full0, const double* pF0,
                    int nF1, const int64_t* full1, const double* pF1, double* out) {
  std::vector<int64_t> pw(R);
  int64_t x = 1;
  for (int r = 0; r < R; ++r) { pw[r] = x; x *= (K + 2); }
  const int S = 1 << R;
  auto lists = [&](const int32_t* lab, int nl, int nF, const int64_t* full,
                   std::vector<int32_t>& lid, std::vector<uint8_t>& lg, std::vector<int64_t>& ptr) {
    std::vector<int> bestg(nl, 255);
    std::vector<int32_t> touched;
    ptr.assign(nF + 1, 0);
    for (int f = 0; f < nF; ++f) {
      const int64_t code = full[f];
      touched.clear();
      for (int s = 0; s < S; ++s) {
        int64_t part = 0;
        int pc = 0;
        for (int r = 0; r < R; ++r)
          if (s >> r & 1) { part += ((code / pw[r]) % (K + 2)) * pw[r]; ++pc; }
        const int32_t l = lab[part];
        if (bestg[l] == 255) touched.push_back(l);
        if (pc < bestg[l]) bestg[l] = pc;
      }
      for (int32_t l : touched) { lid.push_back(l); lg.push_back((uint8_t)bestg[l]); bestg[l] = 255; }
      ptr[f + 1] = (int64_t)lid.size();
    }
  };
  std::vector<int32_t> id0, id1;
  std::vector<uint8_t> g0, g1;
  std::vector<int64_t> p0, p1;
  lists(lab0, nl0, nF0, full0, id0, g0, p0);
  lists(lab1, nl1, nF1, full1, id1, g1, p1);
  // s1[f1][l0] only for labels l0 that appear in some list of column 0
  std::vector<int32_t> used0;
  std::vector<int32_t> pos0(nl0, -1);
  for (int32_t l : id0) if (pos0[l] < 0) { pos0[l] = (int32_t)used0.size(); used0.push_back(l); }
  const int nu = (int)used0.size();
  std::vector<uint8_t> s1((size_t)nF1 * nu, 255);
  for (int f = 0; f < nF1; ++f) {
    uint8_t* row = &s1[(size_t)f * nu];
    for (int64_t j = p1[f]; j < p1[f + 1]; ++j) {
      const int32_t l1 = id1[j];
      const uint8_t g = g1[j];
      for (int u = 0; u < nu; ++u)
        if (g < row[u] && stopT[(int64_t)used0[u] * nl1 + l1]) row[u] = g;
    }
  }
  double tot = 0.0;
  int worst = 0;
  for (int f0 = 0; f0 < nF0; ++f0) {
    double acc = 0.0;
    for (int f1 = 0; f1 < nF1; ++f1) {
      const uint8_t* row = &s1[(size_t)f1 * nu];
      int best = 1 << 20;
      for (int64_t j = p0[f0]; j < p0[f0 + 1]; ++j) {
        const int v = g0[j] + row[pos0[id0[j]]];
        if (v < best) best = v;
      }
      acc += pF1[f1] * best;
      if (best > worst) worst = best;
    }
    tot += pF0[f0] * acc;
  }
  out[0] = tot;
  out[1] = worst;
  out[2] = (double)id0.size() / nF0;
  out[3] = nu;
  return 0;
}

}  // extern "C"

// ------------------------------------------------------------------- decision split (analysis)
// Exact branch and bound (tau = 0; pass fz arrays of -1 so every state keeps its chosen action),
// then a walk of the chosen optimal policy that splits expected reads by the final certificate
// decision.  decT[l0 * nl1 + l1] = 2 if the decision is True, 1 if False, 0 if undecided.
// out[0] = V, out[1] = P(True), out[2] = E[reads * 1{True}], out[3] = E[reads * 1{False}],
// out[4] = 1 if the walk was complete (no overflow / missing action).
extern "C" int bnb_split(int R, int K, int64_t N, const int32_t* lab0, const int32_t* lab1, int nl1,
                         const uint8_t* stopT, const double* m0, const double* m1, const int8_t* dig,
                         const double* W0, const float* F1, int nA, const double* W1, const float* F0,
                         int nB, const int32_t* fz0, const int32_t* fz1, const uint8_t* decT,
                         int64_t max_entries, double* out) {
  BnB S{make(R, K, N, lab0, lab1, nl1, stopT, m0, m1, dig), W0, W1, F0, F1, nA, nB, fz0, fz1,
        nullptr, nullptr, {}, 0, 0, 0, max_entries, false};
  int ml = 22;
  while ((double)(1ULL << ml) * 0.7 < (double)max_entries && ml < 31) ++ml;
  S.memo.maxlog = ml;
  S.memo.init(22);
  out[0] = S.solve(0, INF);
  out[4] = S.overflow ? 0.0 : 1.0;
  if (S.overflow) return 0;
  struct Acc { double a, x, y; };
  std::vector<std::pair<int64_t, Acc>> dummy;
  Memo ma, mx, my;
  ma.init(20); mx.init(20); my.init(20);
  bool ok = true;
  struct Walk {
    BnB& S; const uint8_t* decT; int nl1; Memo &ma, &mx, &my; bool& ok;
    Acc go(int64_t code) {
      const int64_t o0 = code % S.P.N, o1 = code / S.P.N;
      if (S.P.stop(o0, o1)) {
        const uint8_t d = decT[(int64_t)S.P.lab0[o0] * nl1 + S.P.lab1[o1]];
        return Acc{d == 2 ? 1.0 : 0.0, 0.0, 0.0};
      }
      const int64_t j = ma.find((uint64_t)code);
      if (j >= 0) return Acc{ma.vals[j], mx.vals[mx.find((uint64_t)code)], my.vals[my.find((uint64_t)code)]};
      const int64_t i = S.memo.find((uint64_t)code);
      if (i < 0 || !(S.memo.keys[i] & Memo::EXACT) || S.memo.acts[i] < 0) { ok = false; return Acc{0, 0, 0}; }
      BnB::Child ch[8];
      S.raw_children(code, S.memo.acts[i], ch);
      Acc r{0, 0, 0};
      for (int v = 0; v <= S.P.K; ++v) {
        const Acc c = go(ch[v].code);
        r.a += ch[v].p * c.a;
        r.x += ch[v].p * (c.a + c.x);
        r.y += ch[v].p * ((1.0 - c.a) + c.y);
      }
      ma.put((uint64_t)code, r.a, true);
      mx.put((uint64_t)code, r.x, true);
      my.put((uint64_t)code, r.y, true);
      return r;
    }
  } w{S, decT, nl1, ma, mx, my, ok};
  const Acc r = w.go(0);
  out[1] = r.a; out[2] = r.x; out[3] = r.y;
  out[4] = ok ? 1.0 : 0.0;
  return 0;
}
