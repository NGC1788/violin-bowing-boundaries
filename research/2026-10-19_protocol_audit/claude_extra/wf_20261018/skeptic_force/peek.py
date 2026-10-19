import numpy as np, sys
A=sys.argv[1]; idxs=[int(x) for x in sys.argv[2:]]
Z=np.load(f"/Users/jeongjihoon/violin-rne/claude_extra/blocks/blocks_{A}.npz")
D=Z["data"]; fl=list(Z["fields"]); fo=Z["folder"]; tr=Z["trial"]; be=Z["beta"]
np.set_printoptions(linewidth=250, precision=3, suppress=True)
for i in idxs:
    print(f"--- {fo[i]} trial {tr[i]} beta {be[i]:.3f}")
    for k in ["speed","position","force_dc","bridge_dc","nut_dc","bridge_p2p"]:
        x=D[i,fl.index(k)]; n=np.sum(np.isfinite(x))
        print(k, n, x[:n:3])
