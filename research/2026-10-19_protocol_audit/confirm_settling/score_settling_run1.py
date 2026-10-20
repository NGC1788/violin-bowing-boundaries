"""Once-only scorer for the settling prediction (PROTOCOL.md). Claude, 2026-10-20."""
import argparse, csv, datetime as dt, gzip, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HOME = Path.home(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME / "violin-rne/novelty/session_variance"))
import session_variance as SV  # noqa: E402  (rows, references, widen, offset_fit)
TL, TL2, C = SV.TL, SV.TL2, SV.C
STREAM = HOME / "violin-rne/astra_20261019/highforce_stream/2024-03-25_TypeA_sample1_coldwarm"
H = {"upper": 0.035, "lower": 0.079}
CODE = [HERE / "PROTOCOL.md", Path(__file__).resolve(), HOME / "violin-rne/novelty/session_variance/session_variance.py",
        HOME / "violin-rne/novelty/thermal_upper/FROZEN_v2.json", HOME / "violin-rne/claude_extra/verify_astra_boundaries.py",
        HOME / "violin-bowing-boundaries/scripts/regime_map.py", HOME / "violin-rne/final/compare.py"]
KST = dt.timezone(dt.timedelta(hours=9))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_excl(p, o):
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(o, fh, indent=1, default=float)


def coldwarm_rows(V):
    win, beta, speed = defaultdict(dict), {}, defaultdict(list)
    with gzip.open(STREAM / "features.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"] or r["window_s"] != "0.2" or round(float(r["t"]), 2) != 0.2:
                continue
            fpp = r["flybacks_per_period"]
            lab = V.rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                                int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, V.APPLIED)
            k = (r["folder"], int(r["trial"])); win[k] = (float(r["force"]), lab == "helmholtz"); beta[k] = float(r["beta"]); speed[r["folder"]].append(float(r["speed"]))
    out = {}
    for folder in sorted(speed):
        items = sorted((beta[k], k) for k in beta if k[0] == folder)
        assert len(items) == 2000, (folder, len(items))
        lbm = np.log([b for b, _ in items]).reshape(40, 50)
        assert np.all(lbm[1:, 0] > lbm[:-1, -1])
        v = float(np.median(speed[folder])); rows = []
        for c in range(40):
            ks = [k for _, k in items[c * 50:(c + 1) * 50]]
            g = sorted((win[k][0], win[k][1]) for k in ks)
            f = np.log(np.maximum([x[0] for x in g], 1e-3)); h = [x[1] for x in g]
            runs = [i for i in range(len(h) - 1) if h[i] and h[i + 1]]
            if runs:
                s, e = runs[0], runs[-1] + 1
                lower = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s]); upper = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
            else:
                lower, upper = (f[-1], np.inf), (np.nan, np.nan)
            rows.append(dict(lv=np.log(v), lb=float(np.median(lbm[c])), lower=lower, upper=upper))
        out["COLD" if "cold" in folder else "WARM"] = rows
    return out


def a1_main_v2():
    rs = SV.rows_of("A1")
    return [r for r in rs if abs(np.exp(r["lv"]) - 0.10) < 1e-3]


def diff(rA, rB, which, nb=2000, seed=53):
    """level(B) - level(A) with beta columns resampled jointly (keyed by rounded lb)."""
    g = "A T1"
    lA, _, _ = SV.level(rA, which, g); lB, _, _ = SV.level(rB, which, g)
    kA = defaultdict(list); kB = defaultdict(list)
    for r in rA: kA[round(r["lb"], 2)].append(r)
    for r in rB: kB[round(r["lb"], 2)].append(r)
    keys = sorted(set(kA) & set(kB)); rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        kk = rng.choice(len(keys), len(keys))
        a = [r for k in kk for r in kA[keys[k]]]; b = [r for k in kk for r in kB[keys[k]]]
        bs.append(SV.level(b, which, g)[0] - SV.level(a, which, g)[0])
    ci = np.percentile(bs, [2.5, 97.5]).tolist(); d = lB - lA; h = H[which]
    dec = "matters" if (ci[0] > 0 or ci[1] < 0) and abs(d) >= h else ("negligible" if -h < ci[0] and ci[1] < h else "not decided")
    return dict(level_A=lA, level_B=lB, diff=d, ci=ci, threshold=h, decision=dec, n_keys=len(keys))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", action="store_true"); ap.add_argument("--score", action="store_true"); a = ap.parse_args()
    if a.freeze:
        assert not STREAM.exists(), "coldwarm already processed: cannot freeze as a prospective test"
        write_excl(HERE / "FREEZE_RECEIPT.json", dict(frozen_kst=dt.datetime.now(KST).isoformat(), sha256={str(p): sha(p) for p in CODE},
                                                      coldwarm_processed_at_freeze=False)); print("frozen"); return
    rec = json.loads((HERE / "FREEZE_RECEIPT.json").read_text())
    for p in CODE:
        assert sha(p) == rec["sha256"][str(p)], ("changed", p)
    sr = json.loads((STREAM / "receipt.json").read_text()); assert sr["status"] == "validated" and sha(STREAM / "features.csv.gz") == sr["outputs"]["features.csv.gz"]
    write_excl(HERE / "SCORE_CLAIM.json", dict(claimed_kst=dt.datetime.now(KST).isoformat(), receipt_sha256=sha(HERE / "FREEZE_RECEIPT.json")))
    sys.path.insert(0, str(HOME / "violin-rne/claude_extra")); import verify_astra_boundaries as V
    cw = coldwarm_rows(V); main_v2 = a1_main_v2()
    res = dict(primary={w: diff(cw["COLD"], cw["WARM"], w) for w in ("upper", "lower")},
               secondary_day={w: diff(cw["WARM"], main_v2, w) for w in ("upper", "lower")},
               counts={k: dict(upper_finite=sum(np.all(np.isfinite(r["upper"])) for r in v), lower_finite=sum(np.all(np.isfinite(r["lower"])) for r in v),
                               no_band=sum(not np.isfinite(r["upper"][0]) and not np.isfinite(r["upper"][1]) for r in v)) for k, v in list(cw.items()) + [("A1main_v2", main_v2)]},
               finished_kst=dt.datetime.now(KST).isoformat())
    write_excl(HERE / "score_result.json", res); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
