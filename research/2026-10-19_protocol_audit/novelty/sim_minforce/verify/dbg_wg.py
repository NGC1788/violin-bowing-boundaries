import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, vsim as V
T0 = 1000
for loss in ("none", "dist", "bridge"):
    for b in (0.1, 0.05):
        Fs = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
        odv, of, fs = V.wg_run(b, Fs, loss=loss, start="ideal", win=(0.0, 0.4))
        for i, F in enumerate(Fs):
            slip = np.abs(odv[i]) > 1e-3
            nper = slip.size // T0
            per = [int(np.sum(~slip[k*T0:(k+1)*T0-1] & slip[k*T0+1:(k+1)*T0])) for k in range(nper)]
            ss = [int(slip[k*T0:(k+1)*T0].sum()) for k in range(nper)]
            st = odv[i] == 0
            first_bad = next((k for k, c in enumerate(per) if c != 1), None)
            print(f"{loss:6s} b={b} F={F}: first period !=1 slip: {first_bad}; slips/period p0-5 {per[:6]} p20-25 {per[20:26]} p35-39 {per[35:40]}; slip samples p0 {ss[0]} p30 {ss[30]} (ideal {int(b*T0)}); stick f/F range p0 [{of[i][:T0][st[:T0]].min()/F:.4f},{of[i][:T0][st[:T0]].max()/F:.4f}] p30 [{of[i][30*T0:31*T0][st[30*T0:31*T0]].min()/F:.4f},{of[i][30*T0:31*T0][st[30*T0:31*T0]].max()/F:.4f}]", flush=True)
