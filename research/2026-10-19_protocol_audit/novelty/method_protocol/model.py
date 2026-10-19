"""PACE: Protocol-Aware Censored Estimator of the lower (minimum-force) boundary.
Claude, 2026-10-19 (post hoc development on 11 exposed diagrams; see README.md).

Latent law (string physics, measured t=0.2 s force axis):
    log B_dsj = c_s + b log(beta_j) + x_d' eta (properties) + z_ds' gamma (session protocol) + u_d + w_ds + e_dsj
    u_d ~ N(0, tau^2) diagram, w_ds ~ N(0, omega^2) speed-session, e ~ N(0, sigma^2) column.
Observation (session protocol, label free):
    * the column's measured force grid has a floor; a K=2 run that starts at the m lowest strokes
      (s <= m-1) only says "boundary <= hi" -> re-censored to (-inf, hi] when fitting (floor-aware).
    * a point prediction cannot lie below the column's own measured floor:
      p = max(mu, log F_(2nd lowest)) (clip), computed from the target's controls only.
    * zero-width / clamp intervals: measurement-resolution widening by delta (N) [post hoc D_T2 fix].
Fitting: stage 1 = interval-censored ML (session intercepts + common beta slope + sigma, analytic gradient);
stage 2 = REML linear mixed model of session intercepts on speed dummies + covariates.
"""
import numpy as np

import common as C

PROTO_KEYS = ("dec5_bar", "dec5_within", "cal_bar")
PROP_KEYS = ("logZ", "logEI")
SETS = {"mean": (), "props": PROP_KEYS, "propsZ": ("logZ",), "protocol": PROTO_KEYS,
        "protocol_dec5": ("dec5_bar", "dec5_within"), "full": PROP_KEYS + PROTO_KEYS}


def session_table(d, group):
    """Label-free covariates for the 3 speed sessions of one diagram."""
    cov = C.session_covariates(d)
    z, ei = C.PROPS[group]
    out = []
    for f in range(3):
        out.append(dict(folder=f, logZ=np.log(z), logEI=np.log(ei), dec5_bar=cov["diagram"]["dec5"],
                        dec5_within=cov[f]["dec5"] - cov["diagram"]["dec5"], cal_bar=cov["diagram"]["cal"],
                        cal=cov[f]["cal"], dec5=cov[f]["dec5"]))
    return out


def recensor(rows, m):
    """Floor-aware censoring: runs starting at the m lowest strokes -> (-inf, hi]."""
    out = []
    for r in rows:
        r = dict(r)
        if r["band"] and r["s"] is not None and r["s"] <= m - 1:
            r["lo"] = -np.inf
        out.append(r)
    return out


LB0 = -2.76   # centre of log(beta) range (-3.92 .. -1.61)


def shape_features(name, r):
    lb, f = r["lb"], r["folder"]
    e = np.eye(3)[f]
    if name == "lin":
        return np.array([lb])
    if name == "quad":
        return np.array([lb, (lb - LB0) ** 2])
    if name == "cubic":
        return np.array([lb, (lb - LB0) ** 2, (lb - LB0) ** 3])
    if name == "quad_s":
        return np.r_[lb * e, (lb - LB0) ** 2 * e]
    raise ValueError(name)


class PACE:
    def __init__(self, covset="full", m=3, clip=True, delta=0.01, axis="F_ev", shape="lin"):
        self.covset, self.m, self.clip, self.delta, self.axis, self.shape = covset, m, clip, delta, axis, shape
        self.keys = SETS[covset]

    def name(self):
        return f"{self.covset}|m{self.m}|clip{int(self.clip)}" + ("" if self.shape == "lin" else f"|{self.shape}")

    def fit(self, data):
        """data: {tag: (d, group)} exposed training diagrams."""
        rows = {t: recensor(C.intervals(d, self.axis, "H2", "delta", self.delta), self.m) for t, (d, g) in data.items()}
        s1 = C.fit_sessions(rows, lambda r: shape_features(self.shape, r))
        tabs = {t: session_table(d, g) for t, (d, g) in data.items()}
        X, y, grp, se = [], [], [], []
        for (t, f), a, e in zip(s1["keys"], s1["alpha"], s1["se"]):
            row = tabs[t][f]
            X.append(np.r_[np.eye(3)[f], [row[k] for k in self.keys]]); y.append(a); grp.append(t); se.append(e)
        X = np.array(X)
        s2 = C.reml_fit(np.array(y), X, grp, np.array(se))
        self.s1, self.s2 = s1, s2
        self.train_ranges = {k: [float(X[:, 3 + i].min()), float(X[:, 3 + i].max())] for i, k in enumerate(self.keys)}
        return self

    def params(self):
        return dict(name=self.name(), covset=self.covset, keys=list(self.keys), m=self.m, clip=self.clip, delta=self.delta,
                    axis=self.axis, shape=self.shape, theta=self.s1["theta"].tolist(),
                    b=self.s1["b"], sigma=self.s1["sigma"], beta=self.s2["beta"].tolist(),
                    cov=self.s2["cov"].tolist(), tau=self.s2["tau"], omega=self.s2["omega"],
                    tau_at_bound=self.s2["tau_at_bound"], omega_at_bound=self.s2["omega_at_bound"],
                    stage1_success=self.s1["success"], stage2_success=self.s2["success"],
                    train_ranges=self.train_ranges)


def predict_from_params(P, d, group):
    """Column predictions for a target from frozen params P and the target's label-free controls d."""
    tab = session_table(d, group)
    rows = C.intervals(d, P["axis"], None)       # no labels: geometry + floors only
    beta = np.array(P["beta"]); cov = np.array(P["cov"])
    out, extrap = [], {}
    for k in P["keys"]:
        v = tab[0][k] if k != "dec5_within" else None
        if k == "dec5_within":
            vals = [tab[f][k] for f in range(3)]
        else:
            vals = [v]
        lo_, hi_ = P["train_ranges"][k]
        extrap[k] = bool(any(x < lo_ - 1e-12 or x > hi_ + 1e-12 for x in vals))
    for r in rows:
        x = np.r_[np.eye(3)[r["folder"]], [tab[r["folder"]][k] for k in P["keys"]]]
        mu = float(x @ beta + shape_features(P.get("shape", "lin"), r) @ np.array(P.get("theta", [P["b"]])))
        sd = float(np.sqrt(P["sigma"] ** 2 + P["tau"] ** 2 + P["omega"] ** 2 + x @ cov @ x))
        p = max(mu, r["floor1"]) if P["clip"] else mu
        out.append(dict(folder=r["folder"], column=r["column"], lv=r["lv"], lb=r["lb"], mu_latent=mu, point=p, sd=sd,
                        floor1=r["floor1"]))
    level = {f: float(np.r_[np.eye(3)[f], [tab[f][k] for k in P["keys"]]] @ beta) for f in range(3)}
    return out, dict(extrapolated=extrap, session_intercepts=level, covariates=tab)


def lookup_fit(data, delta=0.01):
    """Model-free baseline: per (speed, column) finite midpoints of every training diagram."""
    table = {}
    for t, (d, g) in data.items():
        for r in C.intervals(d, "F_ev", "H2", "delta", delta):
            if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
                table.setdefault((r["folder"], r["column"]), []).append((t, 0.5 * (r["lo"] + r["hi"])))
    # predictive sd: leave-one-training-diagram-out residual scale
    res = []
    tags = list(data)
    for (f, c), lst in table.items():
        for t, mval in lst:
            others = [v for (ff, cc), l2 in table.items() if ff == f and abs(cc - c) <= 2 for tt, v in l2 if tt != t]
            if others:
                res.append(mval - np.median(others))
    sd = float(1.4826 * np.median(np.abs(res))) if res else 0.5
    return dict(table={f"{f},{c}": lst for (f, c), lst in table.items()}, sd=sd, tags=tags)


def lookup_predict(L, d):
    rows = C.intervals(d, "F_ev", None)
    out = []
    for r in rows:
        vals = [v for c in range(r["column"] - 2, r["column"] + 3) for t, v in L["table"].get(f"{r['folder']},{c}", [])]
        p = float(np.median(vals)) if vals else np.nan
        out.append(dict(folder=r["folder"], column=r["column"], lv=r["lv"], lb=r["lb"], point=p, sd=L["sd"]))
    return out


# ----------------------------------------------------------------------------------------- scoring
def score(rows, preds, delta=0.01):
    """Column-level scores of one target diagram against its (delta-fixed) observed lower intervals.
    rows: C.intervals(d, 'F_ev', 'H2', 'delta') ; preds: list aligned by (folder, column)."""
    P = {(p["folder"], p["column"]): p for p in preds}
    fin_err, dist, ls, hits, n_avail, n_pred_fail = [], [], [], 0, 0, 0
    mu0, sc_rows = [], []
    for r in rows:
        if np.isnan(r["lo"]) and np.isnan(r["hi"]):
            continue
        p = P[(r["folder"], r["column"])]
        n_avail += 1
        if not np.isfinite(p["point"]):
            n_pred_fail += 1
            continue
        x = p["point"]
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            fin_err.append(abs(x - 0.5 * (r["lo"] + r["hi"])))
        dist.append(max(r["lo"] - x, x - r["hi"], 0.0))
        hits += int(r["lo"] <= x <= r["hi"])
        ls.append(float(C.log_int_prob(np.array([r["lo"]]), np.array([r["hi"]]), np.array([x]), p["sd"])[0]))
        mu0.append(x); sc_rows.append(r)
    lev = C.fit_offset(sc_rows, np.array(mu0)) if sc_rows else dict(success=False)
    return dict(n_avail=n_avail, n_pred_fail=n_pred_fail, n_finite=len(fin_err),
                col_median_abs_err=float(np.median(fin_err)) if fin_err else np.nan,
                col_mean_abs_err=float(np.mean(fin_err)) if fin_err else np.nan,
                interval_distance=float(np.mean(dist)) if dist else np.nan,
                hit_rate=hits / n_avail if n_avail else np.nan,
                log_score=float(np.mean(ls)) if ls else np.nan,
                level_error=abs(lev["level"]) if lev.get("success") else np.nan,
                level_signed=lev.get("level"), fin_errors=fin_err)


class LookupLevel:
    """Model-free shape (same-speed, +-2 beta-column lookup of training finite midpoints) + optional
    label-free level regression of the training diagrams' offsets (inner leave-one-out lookups) + floor clip."""
    def __init__(self, keys=(), clip=True, delta=0.01):
        self.keys, self.clip, self.delta = tuple(keys), clip, delta

    def name(self):
        return "lookup" + ("+" + "+".join(self.keys) if self.keys else "") + ("|clip1" if self.clip else "")

    def fit(self, data):
        self.L = lookup_fit(data, self.delta)
        self.offsets = {}
        if self.keys:
            X, y = [], []
            for t, (d, g) in data.items():
                Li = lookup_fit({u: v for u, v in data.items() if u != t}, self.delta)
                pr = lookup_predict(Li, d)
                rows = C.intervals(d, "F_ev", "H2", "delta", self.delta)
                ok = [i for i, p in enumerate(pr) if np.isfinite(p["point"])]
                fo = C.fit_offset([rows[i] for i in ok], np.array([pr[i]["point"] for i in ok]))
                tab = session_table(d, g)[0]
                X.append(np.r_[1.0, [tab[k] for k in self.keys]]); y.append(fo["level"])
                self.offsets[t] = fo["level"]
            X, y = np.array(X), np.array(y)
            self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
            self.train_ranges = {k: [float(X[:, 1 + i].min()), float(X[:, 1 + i].max())] for i, k in enumerate(self.keys)}
        return self

    def predict(self, d, group):
        pr = lookup_predict(self.L, d)
        shift, extrap = 0.0, {}
        if self.keys:
            tab = session_table(d, group)[0]
            shift = float(np.r_[1.0, [tab[k] for k in self.keys]] @ self.coef)
            extrap = {k: not (self.train_ranges[k][0] <= tab[k] <= self.train_ranges[k][1]) for k in self.keys}
        rows = C.intervals(d, "F_ev", None)
        fl = {(r["folder"], r["column"]): r["floor1"] for r in rows}
        for p in pr:
            p["point"] = p["point"] + shift
            if self.clip and np.isfinite(p["point"]):
                p["point"] = max(p["point"], fl[(p["folder"], p["column"])])
        return pr, dict(extrapolated=extrap, shift=shift)
