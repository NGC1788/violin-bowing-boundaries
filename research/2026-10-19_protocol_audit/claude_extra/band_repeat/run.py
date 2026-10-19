"""Run once after PLAN.md is hashed (Claude, 2026-10-19). Writes result.json with O_EXCL."""
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "violin-rne/astra_20261019/observation_model"))
from band_probe import posterior  # noqa: E402

CACHE = Path.home() / "violin-rne/final/cache"


def cols(tag, t=0.2):
    d = dict(np.load(CACHE / f"{tag}.npz"))
    g = defaultdict(list)
    for i in np.flatnonzero(np.isclose(d["t"], t)):
        g[(round(float(d["speed"][i]), 2), int(d["column"][i]))].append(i)
    out = {}
    for k, idx in g.items():
        idx = sorted(idx, key=lambda i: d["force"][i])
        out[k] = (np.log(np.maximum(d["force"][idx], 1e-3)), d["H"][idx].astype(int))
    return out


def krule(y, K):
    runs = [i for i in range(len(y) - K + 1) if all(y[i:i + K])]
    return (None, None) if not runs else (runs[0], runs[-1] + K)


def loc(f, b):
    return None if b is None or b <= 0 or b >= len(f) else 0.5 * (f[b - 1] + f[b])


def estimates(f, y):
    p = posterior(y, integrated=True)
    return {"K2": krule(y, 2), "K3": krule(y, 3), "BAND": tuple(p["median"]) if p["median"][0] < p["median"][1] else (None, None)}


def main():
    plan_sha = hashlib.sha256((HERE / "PLAN.md").read_bytes()).hexdigest()
    rec = json.loads((HERE / "freeze.json").read_text())
    assert rec["plan_sha256"] == plan_sha and rec["code_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    c1, c2 = cols("B_T2r1"), cols("B_T2r2")
    keys = sorted(set(c1) & set(c2))
    rows = []
    for k in keys:
        e1, e2 = estimates(*c1[k]), estimates(*c2[k])
        r = dict(key=k)
        for m in ("K2", "K3", "BAND"):
            for w, j in (("lower", 0), ("upper", 1)):
                r[(m, w)] = (loc(c1[k][0], e1[m][j]), loc(c2[k][0], e2[m][j]))
        rows.append(r)
    rng = np.random.default_rng(20261019)
    betas = sorted({k[1] for k in keys})
    out = dict(plan_sha256=plan_sha, n_pairs=len(keys), boundaries={})
    for w in ("lower", "upper"):
        common = [r for r in rows if all(None not in r[(m, w)] for m in ("K2", "K3", "BAND"))]
        med = lambda rs, m: float(np.median([abs(r[(m, w)][0] - r[(m, w)][1]) for r in rs]))
        res = dict(n_common=len(common), common={m: med(common, m) for m in ("K2", "K3", "BAND")})
        own = {}
        for m in ("K2", "K3", "BAND"):
            rs = [r for r in rows if None not in r[(m, w)]]
            own[m] = dict(n=len(rs), median_abs_diff=med(rs, m) if rs else None)
        res["own_set"] = own
        diffs = []
        for _ in range(2000):
            bb = rng.choice(betas, len(betas))
            rs = [r for b in bb for r in common if r["key"][1] == b]
            if rs:
                diffs.append(med(rs, "BAND") - med(rs, "K2"))
        res["band_minus_k2"] = res["common"]["BAND"] - res["common"]["K2"]
        res["band_minus_k2_ci"] = np.percentile(diffs, [2.5, 97.5]).tolist()
        res["band_offset_vs_k2_signed_median"] = float(np.median([r[("BAND", w)][i] - r[("K2", w)][i] for r in common for i in (0, 1)]))
        out["boundaries"][w] = res
    lo, up = out["boundaries"]["lower"], out["boundaries"]["upper"]
    better_both = lo["band_minus_k2"] < 0 and up["band_minus_k2"] < 0
    ci_one = (lo["band_minus_k2_ci"][1] < 0) or (up["band_minus_k2_ci"][1] < 0)
    out["decision"] = "consistent improvement" if (better_both and ci_one) else "no consistent improvement -> method direction ends"
    fd = os.open(HERE / "result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
