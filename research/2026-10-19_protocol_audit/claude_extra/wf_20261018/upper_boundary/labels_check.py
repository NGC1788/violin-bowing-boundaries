"""Physical checks of the labels around the upper boundary (A1, A2; t = 0.2 s, 0.2 s windows).
 - flyback law: for Helmholtz motion the bridge-force flyback height ~ 2 Z v / beta, so Z_eff = h beta / (2 v)
   should be the same just below the upper boundary as deep inside the region.  Also computed for non-Helmholtz
   windows above the boundary (are they Helmholtz-like waveforms mislabelled by the flyback count?).
 - pitch flattening: f0 (cents re 98.08 Hz) of Helmholtz windows inside vs just below the upper boundary.
 - periodicity and second-flyback ratio of windows just above the boundary.
"""
import numpy as np

import ub

T = 0.2


def main():
    for tag in ("A1", "A2"):
        sp = ub.Spec(tag)
        w = sp.w
        groups = ub.column_groups(sp, T)
        print(f"=== {tag}, t = {T}")
        for f in range(3):
            v = sp.speed[f]
            inside, below, above = [], [], []
            for k, g in groups.items():
                if k[0] != f:
                    continue
                fs = [x[0] for x in g]; hs = [x[1] for x in g]
                lo_u, hi_u, st_u = ub.upper_interval(fs, hs)
                lo_l, hi_l, st_l = ub.lower_interval(fs, hs)
                if st_u != "interval" or st_l not in ("interval", "left"):
                    continue
                lf = np.log(np.maximum(fs, 1e-4))
                e = int(np.searchsorted(lf, hi_u))          # first stroke above the upper boundary
                s0 = 0 if st_l == "left" else int(np.searchsorted(lf, hi_l))  # first stroke of the region
                rows = [x for x in g]
                for j in range(s0 + 3, e - 3):
                    if rows[j][1]:
                        inside.append(rows[j])
                for j in range(max(s0, e - 3), e):
                    if rows[j][1]:
                        below.append(rows[j])
                for j in range(e, min(len(rows), e + 3)):
                    above.append(rows[j])
            d = sp.at(T)

            def stats(lst, only_label=None):
                idx = [d[x[4]][2] for x in lst if only_label is None or x[2] == only_label]
                if not idx:
                    return None
                idx = np.array(idx)
                z = w["fh"][idx] * w["beta"][idx] / (2 * v)
                cents = 1200 * np.log2(w["f0"][idx] / 98.08)
                return dict(n=len(idx), Z=np.nanmedian(z), Zq=np.nanpercentile(z, [25, 75]), cents=np.nanmedian(cents),
                            per=np.nanmedian(w["per"][idx]), sfr=np.nanmedian(w["sfr"][idx]))
            si, sb = stats(inside), stats(below)
            print(f"  v{f+1}: Helmholtz inside region: n {si['n']}, Z_eff {si['Z']:.3f} kg/s (IQR {si['Zq'][0]:.3f}-{si['Zq'][1]:.3f}), "
                  f"f0 {si['cents']:+.1f} cents, periodicity {si['per']:.3f}, second-flyback ratio {si['sfr']:.2f}")
            print(f"       Helmholtz just below upper bd: n {sb['n']}, Z_eff {sb['Z']:.3f} (IQR {sb['Zq'][0]:.3f}-{sb['Zq'][1]:.3f}), "
                  f"f0 {sb['cents']:+.1f} cents, periodicity {sb['per']:.3f}, second-flyback ratio {sb['sfr']:.2f}")
            for li, name in enumerate(ub.LABELS):
                sa = stats(above, li)
                if sa and sa["n"] >= 5:
                    print(f"       above upper bd, {name:14s}: n {sa['n']:3d}, Z_eff {sa['Z']:.3f} (IQR {sa['Zq'][0]:.3f}-{sa['Zq'][1]:.3f}), "
                          f"f0 {sa['cents']:+.1f} cents, periodicity {sa['per']:.3f}, second-flyback ratio {sa['sfr']:.2f}")


if __name__ == "__main__":
    main()
