// Exact solvers for optimal_c3.py (C=3 columns).  Compiled at run time by optimal_c3.py.
//
// A joint state is code = o0 + N*o1 + N*N*o2, where o_c = sum_r d_{c,r} B^r is column c's
// partial observation (digit 0 unknown, 1+level otherwise), N = B^R, B = K + 2 (the encoding of
// optimal_scaled_kernel.cpp with a third column).  The frozen certificate depends on a column only
// through its per-column label lab_c[o_c]; tup_c[label][rule][6] holds (any candidate, any invalid
// possible start, min lower, max upper, max lower, min upper) over the candidates, and the stop
// test is the frozen core.certificates() formula over three columns (evaluated on the fly):
//   could_k = all_c cand && max_c minlower <= min_c maxupper
//   guar_k  = all_c !invalid && max_c maxlower <= min_c minupper
//   stop    = any_k guar_k || !any_k could_k.
// The prior is a product over columns, so P(level v at (c, r) | state) = m_c[o_c+(v+1)B^r]/m_c[o_c].
//
// Column-decomposed lower bound: V(o) >= sum_c E[ W_c(o_c | the other two columns fully known) ],
// the expectation over the other two columns' full labels (fa, fb) under their completion
// distributions G_a(. | o_a), G_b(. | o_b); W_c[o_c][cls_c(fa, fb)] is a single-column DP.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

const double INF = std::numeric_limits<double>::infinity();

struct Problem {
  int R, K, B;
  int64_t N;
  const int32_t* lab[3];
  const double* tup[3];
  const double* m[3];
  const int8_t* dig;
  std::vector<int64_t> pw;
  int64_t colstep[3];
  inline void split(int64_t code, int64_t* o) const {
    o[0] = code % N;
    const int64_t q = code / N;
    o[1] = q % N;
    o[2] = q / N;
  }
  inline bool stop_lab(int32_t l0, int32_t l1, int32_t l2) const {
    const double* a = tup[0] + (int64_t)l0 * K * 6;
    const double* b = tup[1] + (int64_t)l1 * K * 6;
    const double* c = tup[2] + (int64_t)l2 * K * 6;
    bool anycould = false;
    for (int k = 0; k < K; ++k, a += 6, b += 6, c += 6) {
      if (a[1] == 0 && b[1] == 0 && c[1] == 0) {
        const double il = std::max(a[4], std::max(b[4], c[4]));
        const double ih = std::min(a[5], std::min(b[5], c[5]));
        if (il <= ih) return true;
      }
      if (a[0] > 0 && b[0] > 0 && c[0] > 0) {
        const double ol = std::max(a[2], std::max(b[2], c[2]));
        const double oh = std::min(a[3], std::min(b[3], c[3]));
        if (ol <= oh) anycould = true;
      }
    }
    return !anycould;
  }
  inline bool stop(const int64_t* o) const { return stop_lab(lab[0][o[0]], lab[1][o[1]], lab[2][o[2]]); }
  inline double marg(const int64_t* o) const { return m[0][o[0]] * m[1][o[1]] * m[2][o[2]]; }
};

Problem make(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
             const double* const* m, const int8_t* dig) {
  Problem p;
  p.R = R; p.K = K; p.B = K + 2; p.N = N; p.dig = dig;
  for (int c = 0; c < 3; ++c) { p.lab[c] = lab[c]; p.tup[c] = tup[c]; p.m[c] = m[c]; }
  p.pw.resize(R);
  int64_t x = 1;
  for (int r = 0; r < R; ++r) { p.pw[r] = x; x *= p.B; }
  p.colstep[0] = 1; p.colstep[1] = N; p.colstep[2] = N * N;
  return p;
}

struct LB {
  // per column c: W[c] (N x nA[c]); cls[c] (nf[other0] x nf[other1]); completion distribution of
  // column c over its full labels as CSR (gptr N+1, gidx, gval); fz[c][o] = full-label index of a
  // frozen column state (its label can never change), else -1.
  const double* W[3];
  int nA[3];
  const int32_t* cls[3];
  int nf[3];
  const int64_t* gptr[3];
  const int32_t* gidx[3];
  const double* gval[3];
  const int32_t* fz[3];
  static inline void others(int c, int& a, int& b) { a = c == 0 ? 1 : 0; b = c == 2 ? 1 : 2; }
  inline double term(int c, const int64_t* o) const {
    int a, b;
    others(c, a, b);
    const double* w = W[c] + (size_t)o[c] * nA[c];
    const int32_t* cl = cls[c];
    const int nb = nf[b];
    double s = 0.0;
    for (int64_t ja = gptr[a][o[a]]; ja < gptr[a][o[a] + 1]; ++ja) {
      const int32_t* row = cl + (size_t)gidx[a][ja] * nb;
      double in = 0.0;
      for (int64_t jb = gptr[b][o[b]]; jb < gptr[b][o[b] + 1]; ++jb) in += gval[b][jb] * w[row[gidx[b][jb]]];
      s += gval[a][ja] * in;
    }
    return s;
  }
  inline double value(const int64_t* o) const {  // o not stopped
    const double s = term(0, o) + term(1, o) + term(2, o);
    return std::max(1.0, s * (1.0 - 1e-6) - 1e-9);  // margin for float rounding of G
  }
  // exact value when at least two columns are frozen: the third faces a single-column problem
  inline bool frozen_value(const int64_t* o, double& v) const {
    const bool f0 = fz[0][o[0]] >= 0, f1 = fz[1][o[1]] >= 0, f2 = fz[2][o[2]] >= 0;
    if ((int)f0 + (int)f1 + (int)f2 < 2) return false;
    int c = !f0 ? 0 : !f1 ? 1 : 2;  // the non-frozen column (or column 2 if all are frozen)
    if (f0 && f1 && f2) c = 2;
    int a, b;
    others(c, a, b);
    v = W[c][(size_t)o[c] * nA[c] + cls[c][(size_t)fz[a][o[a]] * nf[b] + fz[b][o[b]]]];
    return true;
  }
  inline bool col_frozen(int c, const int64_t* o) const { return fz[c][o[c]] >= 0; }
};

struct Memo {
  std::vector<uint64_t> keys;
  std::vector<double> vals;
  std::vector<int8_t> acts;
  uint64_t mask = 0, used = 0;
  int maxlog = 28;
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
    ok.swap(keys); ov.swap(vals); oa.swap(acts);
    const uint64_t cap = (mask + 1) * 2;
    keys.assign(cap, EMPTY); vals.assign(cap, 0.0); acts.assign(cap, -1);
    mask = cap - 1;
    for (size_t j = 0; j < ok.size(); ++j) {
      if (ok[j] == EMPTY) continue;
      uint64_t i = h(ok[j] & ~EXACT) & mask;
      while (keys[i] != EMPTY) i = (i + 1) & mask;
      keys[i] = ok[j]; vals[i] = ov[j]; acts[i] = oa[j];
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

struct Child { int64_t code; double p; double val; bool exact; };
struct Act { double qlb; int cell; };

struct BnB {
  Problem P;
  LB L;
  Memo memo, gmemo, pmemo;
  uint64_t calls = 0, expanded = 0, frozen_hits = 0, truncated = 0;
  int64_t max_entries = 0;
  bool overflow = false, policy_broken = false;
  double tau = 0.0, gcut = 1e-3;
  int trunc_mode = 0;  // 0: truncated value = LB (certified lower run); 1: #unread (upper run)

  inline int unread(const int64_t* o) const {  // unread cells in non-frozen columns
    int u = 0;
    for (int c = 0; c < 3; ++c) {
      if (L.col_frozen(c, o)) continue;
      for (int r = 0; r < P.R; ++r) u += P.dig[o[c] * P.R + r] == 0;
    }
    return u;
  }
  inline bool readable(int cell, const int64_t* o) const {
    const int c = cell / P.R, r = cell % P.R;
    return P.dig[o[c] * P.R + r] == 0 && !L.col_frozen(c, o);
  }
  void raw_children(int64_t code, const int64_t* o, int cell, Child* out) const {
    const int c = cell / P.R, r = cell % P.R;
    const double* m = P.m[c];
    const double inv = 1.0 / m[o[c]];
    for (int v = 0; v <= P.K; ++v) {
      const int64_t step = (int64_t)(v + 1) * P.pw[r];
      out[v].code = code + step * P.colstep[c];
      out[v].p = m[o[c] + step] * inv;
    }
  }
  inline bool trunc_value(const int64_t* o, double& v) {
    if (tau <= 0.0 || P.marg(o) >= tau) return false;
    if (trunc_mode == 0) v = L.value(o);
    else v = (double)unread(o);
    return true;
  }
  inline void child_info(Child& ch) {
    int64_t o[3];
    P.split(ch.code, o);
    if (P.stop(o)) { ch.val = 0.0; ch.exact = true; return; }
    if (L.frozen_value(o, ch.val)) { ch.exact = true; return; }
    if (trunc_value(o, ch.val)) { ch.exact = true; return; }
    const int64_t i = memo.find((uint64_t)ch.code);
    if (i >= 0) { ch.val = memo.vals[i]; ch.exact = (memo.keys[i] & Memo::EXACT) != 0; return; }
    ch.val = L.value(o);
    ch.exact = false;
  }

  // Returns V(code) exactly if V(code) < beta; otherwise a lower bound >= beta.
  double solve(int64_t code, double beta) {
    ++calls;
    int64_t o[3];
    P.split(code, o);
    if (P.stop(o)) return 0.0;
    {
      double fv;
      if (L.frozen_value(o, fv)) { ++frozen_hits; return fv; }
      if (trunc_value(o, fv)) { ++truncated; return fv; }
    }
    double lb0;
    const int64_t i = memo.find((uint64_t)code);
    if (i >= 0) {
      if (memo.keys[i] & Memo::EXACT) return memo.vals[i];
      lb0 = memo.vals[i];
    } else {
      lb0 = L.value(o);
    }
    if (lb0 >= beta) return lb0;
    if ((int64_t)memo.used > max_entries || memo.full) { overflow = true; return lb0; }
    ++expanded;
    const int R = P.R, K = P.K;
    Act acts[96];
    int na = 0;
    Child tmp[8];
    for (int cell = 0; cell < 3 * R; ++cell) {
      if (!readable(cell, o)) continue;
      raw_children(code, o, cell, tmp);
      double q = 1.0;
      for (int v = 0; v <= K; ++v) { child_info(tmp[v]); q += tmp[v].p * tmp[v].val; }
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
      raw_children(code, o, acts[a].cell, ch);
      for (int v = 0; v <= K; ++v) child_info(ch[v]);
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
        const int64_t k = memo.find((uint64_t)c.code);
        int64_t oc[3];
        P.split(c.code, oc);
        double dummy;
        c.exact = ((k >= 0) && (memo.keys[k] & Memo::EXACT)) || L.frozen_value(oc, dummy) ||
                  trunc_value(oc, dummy) || P.stop(oc);
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
    if (found) { memo.put((uint64_t)code, best, true, bestcell); return best; }
    const double v = std::max(lb0, newlb);
    memo.put((uint64_t)code, v, false);
    return v;
  }

  // one-step lookahead on the decomposed bound (frozen columns skipped, >=2 frozen -> exact DP)
  int greedy_cell(int64_t code, const int64_t* o) {
    double bq = INF;
    int bc = -1;
    Child ch[8];
    for (int cell = 0; cell < 3 * P.R; ++cell) {
      if (!readable(cell, o)) continue;
      raw_children(code, o, cell, ch);
      double q = 1.0;
      for (int v = 0; v <= P.K; ++v) {
        int64_t oc[3];
        P.split(ch[v].code, oc);
        double cv;
        if (P.stop(oc)) cv = 0.0;
        else if (!L.frozen_value(oc, cv)) cv = L.value(oc);
        q += ch[v].p * cv;
      }
      if (q < bq - 1e-12) { bq = q; bc = cell; }
    }
    return bc;
  }
  // exact expected reads of the greedy policy from code; below tau*gcut: #unread (an upper bound)
  bool gready = false;
  double greedy_value(int64_t code, bool cut) {
    int64_t o[3];
    P.split(code, o);
    if (P.stop(o)) return 0.0;
    double fv;
    if (L.frozen_value(o, fv)) return fv;
    if (cut && P.marg(o) < tau * gcut) return (double)unread(o);
    if (!gready) { gmemo.init(20); gready = true; }
    const int64_t i = gmemo.find((uint64_t)code);
    if (i >= 0) return gmemo.vals[i];
    if (gmemo.full || (int64_t)gmemo.used > max_entries) { overflow = true; return 0.0; }
    const int bc = greedy_cell(code, o);
    Child ch[8];
    raw_children(code, o, bc, ch);
    double e = 1.0;
    for (int v = 0; v <= P.K; ++v) e += ch[v].p * greedy_value(ch[v].code, cut);
    gmemo.put((uint64_t)code, e, true);
    return e;
  }
  // exact expected reads of the policy the (truncated) search chose; greedy below tau
  bool pready = false;
  double policy_value(int64_t code) {
    int64_t o[3];
    P.split(code, o);
    if (P.stop(o)) return 0.0;
    double fv;
    if (L.frozen_value(o, fv)) return fv;
    if (tau > 0.0 && P.marg(o) < tau) return greedy_value(code, true);
    if (!pready) { pmemo.init(20); pready = true; }
    const int64_t j = pmemo.find((uint64_t)code);
    if (j >= 0) return pmemo.vals[j];
    const int64_t i = memo.find((uint64_t)code);
    if (i < 0 || !(memo.keys[i] & Memo::EXACT) || memo.acts[i] < 0) { policy_broken = true; return INF; }
    Child ch[8];
    raw_children(code, o, memo.acts[i], ch);
    double e = 1.0;
    for (int v = 0; v <= P.K; ++v) e += ch[v].p * policy_value(ch[v].code);
    pmemo.put((uint64_t)code, e, true);
    return e;
  }
};

void setup(BnB& S, int64_t max_entries) {
  S.max_entries = max_entries;
  int ml = 22;
  while ((double)(1ULL << ml) * 0.7 < (double)max_entries && ml < 31) ++ml;
  S.memo.maxlog = ml;
  S.gmemo.maxlog = ml - 1;
  S.pmemo.maxlog = ml - 1;
  S.memo.init(22);
}

LB make_lb(const double* const* W, const int* nA, const int32_t* const* cls, const int* nf,
           const int64_t* const* gptr, const int32_t* const* gidx, const double* const* gval,
           const int32_t* const* fz) {
  LB L;
  for (int c = 0; c < 3; ++c) {
    L.W[c] = W[c]; L.nA[c] = nA[c]; L.cls[c] = cls[c]; L.nf[c] = nf[c];
    L.gptr[c] = gptr[c]; L.gidx[c] = gidx[c]; L.gval[c] = gval[c]; L.fz[c] = fz[c];
  }
  return L;
}

}  // namespace

extern "C" {

// stop verdicts for a list of joint codes (validation against the frozen certificates())
int stop_codes(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
               const double* const* m, const int8_t* dig, int64_t n, const int64_t* codes,
               uint8_t* out) {
  Problem P = make(R, K, N, lab, tup, m, dig);
  for (int64_t i = 0; i < n; ++i) {
    int64_t o[3];
    P.split(codes[i], o);
    out[i] = P.stop(o);
  }
  return 0;
}

// Full Bellman recursion over all N^3 joint codes (children have larger codes); validation only.
// Frozen columns are NOT skipped here (independent of that shortcut).
// out[0] = optimal expected reads, out[1] = minimax reads, out[2] = non-stopped states.
int dense_solve(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
                const double* const* m, const int8_t* dig, double* out) {
  Problem P = make(R, K, N, lab, tup, m, dig);
  const int64_t NN = N * N * N;
  std::vector<double> V(NN);
  std::vector<uint8_t> MM(NN);
  int64_t nonstop = 0;
  for (int64_t code = NN - 1; code >= 0; --code) {
    int64_t o[3];
    P.split(code, o);
    if (P.stop(o)) { V[code] = 0.0; MM[code] = 0; continue; }
    ++nonstop;
    double best = INF;
    int worst = 1 << 20;
    for (int c = 0; c < 3; ++c) {
      const double inv = 1.0 / P.m[c][o[c]];
      for (int r = 0; r < R; ++r) {
        if (dig[o[c] * R + r]) continue;
        double e = 1.0;
        int wc = 1;
        for (int v = 0; v <= K; ++v) {
          const int64_t step = (int64_t)(v + 1) * P.pw[r];
          const int64_t ch = code + step * P.colstep[c];
          e += P.m[c][o[c] + step] * inv * V[ch];
          wc = std::max(wc, 1 + (int)MM[ch]);
        }
        best = std::min(best, e);
        worst = std::min(worst, wc);
      }
    }
    V[code] = best;
    MM[code] = (uint8_t)std::min(worst, 255);
  }
  out[0] = V[0];
  out[1] = MM[0];
  out[2] = (double)nonstop;
  return 0;
}

// Branch and bound.  out: [0] root value, [1] calls, [2] expanded, [3] overflow, [4] memo size,
// [5] root exact, [6] root LB, [7] frozen hits, [8] truncated, [9] greedy memo, [10] value of the
// policy the search chose (greedy below tau; inf if unavailable), [11] its memo size.
int bnb_solve(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
              const double* const* m, const int8_t* dig, const double* const* W, const int* nA,
              const int32_t* const* cls, const int* nf, const int64_t* const* gptr,
              const int32_t* const* gidx, const double* const* gval, const int32_t* const* fz,
              double ub, int64_t max_entries, double tau, int trunc_mode, double gcut, double* out) {
  BnB S;
  S.P = make(R, K, N, lab, tup, m, dig);
  S.L = make_lb(W, nA, cls, nf, gptr, gidx, gval, fz);
  S.tau = tau;
  S.trunc_mode = trunc_mode;
  S.gcut = gcut;
  setup(S, max_entries);
  const double v = S.solve(0, ub);
  out[0] = v;
  out[1] = (double)S.calls;
  out[2] = (double)S.expanded;
  out[3] = S.overflow ? 1.0 : 0.0;
  out[4] = (double)S.memo.used;
  const int64_t i = S.memo.find(0);
  int64_t o0[3] = {0, 0, 0};
  double dummy;
  out[5] = ((i >= 0 && (S.memo.keys[i] & Memo::EXACT)) || S.P.stop(o0) || S.L.frozen_value(o0, dummy))
               ? 1.0 : 0.0;
  out[6] = S.P.stop(o0) ? 0.0 : S.L.value(o0);
  out[7] = (double)S.frozen_hits;
  out[8] = (double)S.truncated;
  out[10] = INF;
  out[11] = 0.0;
  if (out[5] > 0.5 && !S.overflow && trunc_mode == 0) {
    const double pv = S.policy_value(0);
    out[10] = (S.policy_broken || S.overflow || S.gmemo.full || S.pmemo.full) ? INF : pv;
    out[11] = S.pready ? (double)S.pmemo.used : 0.0;
  }
  out[9] = S.gready ? (double)S.gmemo.used : 0.0;
  return 0;
}

// Exact expected reads of the one-step-lookahead (LB) policy.  out: [0] reads, [1] states,
// [2] overflow.
int greedy_eval(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
                const double* const* m, const int8_t* dig, const double* const* W, const int* nA,
                const int32_t* const* cls, const int* nf, const int64_t* const* gptr,
                const int32_t* const* gidx, const double* const* gval, const int32_t* const* fz,
                int64_t max_entries, double* out) {
  BnB S;
  S.P = make(R, K, N, lab, tup, m, dig);
  S.L = make_lb(W, nA, cls, nf, gptr, gidx, gval, fz);
  setup(S, max_entries);
  out[0] = S.greedy_value(0, false);
  out[1] = S.gready ? (double)S.gmemo.used : 0.0;
  out[2] = (S.overflow || S.gmemo.full) ? 1.0 : 0.0;
  return 0;
}

// lower bound at a list of joint codes (Python policy walks use it at truncated leaves)
int lb_codes(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
             const double* const* m, const int8_t* dig, const double* const* W, const int* nA,
             const int32_t* const* cls, const int* nf, const int64_t* const* gptr,
             const int32_t* const* gidx, const double* const* gval, const int32_t* const* fz,
             int64_t n, const int64_t* codes, double* out) {
  BnB S;
  S.P = make(R, K, N, lab, tup, m, dig);
  S.L = make_lb(W, nA, cls, nf, gptr, gidx, gval, fz);
  for (int64_t i = 0; i < n; ++i) {
    int64_t o[3];
    S.P.split(codes[i], o);
    double v;
    if (S.P.stop(o)) v = 0.0;
    else if (!S.L.frozen_value(o, v)) v = S.L.value(o);
    out[i] = v;
  }
  return 0;
}

// Hindsight: for each given triple of FULL column codes, the fewest cells whose reveal makes the
// certificate stop (subsets of the 3R cells enumerated by increasing size).  labsub_c[f * 2^R + s]
// = label of full code f restricted to the cell subset s.  out[0] = weighted mean, out[1] = max.
int hindsight(int R, int K, int64_t N, const int32_t* const* lab, const double* const* tup,
              const double* const* m, const int8_t* dig, const int32_t* const* labsub, int64_t n,
              const int32_t* f0, const int32_t* f1, const int32_t* f2, const double* w,
              int nmask, const int32_t* masks, const int8_t* popc, double* out) {
  Problem P = make(R, K, N, lab, tup, m, dig);
  const int S = 1 << R;
  const int lo = S - 1;
  double tot = 0.0;
  int worst = 0;
  for (int64_t t = 0; t < n; ++t) {
    const int32_t* a = labsub[0] + (size_t)f0[t] * S;
    const int32_t* b = labsub[1] + (size_t)f1[t] * S;
    const int32_t* c = labsub[2] + (size_t)f2[t] * S;
    int best = 3 * R;
    for (int j = 0; j < nmask; ++j) {
      const int32_t mk = masks[j];
      if (P.stop_lab(a[mk & lo], b[(mk >> R) & lo], c[mk >> (2 * R)])) { best = popc[j]; break; }
    }
    tot += w[t] * best;
    worst = std::max(worst, best);
  }
  out[0] = tot;
  out[1] = worst;
  return 0;
}

}  // extern "C"
