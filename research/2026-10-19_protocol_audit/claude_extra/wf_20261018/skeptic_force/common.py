"""Shared loader for the skeptic force-artifact tests (Claude, 2026-10-18). Reads blocks_<A>.npz (read-only)."""
import numpy as np
from pathlib import Path

BLK = Path.home() / "violin-rne/claude_extra/blocks"
OUT = Path.home() / "violin-rne/claude_extra/wf_20261018/skeptic_force"
SPEED = {"v1": 0.05, "v2": 0.10, "v3": 0.20}


class Spec:
    def __init__(self, A):
        Z = np.load(BLK / f"blocks_{A}.npz", allow_pickle=False)
        self.A = A
        D = Z["data"].astype(float)
        self.fields = [str(x) for x in Z["fields"]]
        for i, n in enumerate(self.fields):
            setattr(self, n, D[:, i, :])
        self.folder = np.array([str(x).split("_")[-1] for x in Z["folder"]])  # v1/v2/v3
        self.trial = Z["trial"].astype(int)
        self.beta = Z["beta"].astype(float)
        self.dt = float(Z["block_s"])
        n, nb = D.shape[0], D.shape[2]
        self.n, self.nb = n, nb
        r = (self.trial - 1) % 50 + 1
        c = (self.trial - 1) // 50
        if A == "A1":
            r = 51 - r
            c = 39 - c
        self.rank, self.col = r, c
        # order of recording within a folder session = trial index (both specimens)
        self.v = np.array([SPEED[f] for f in self.folder])
        sp = np.nan_to_num(self.speed)
        vmax = sp.max(axis=1)
        self.moving = sp > 0.02 * vmax[:, None]
        self.onset = np.argmax(self.moving, axis=1)
        self.last_move = nb - 1 - np.argmax(self.moving[:, ::-1], axis=1)
        self.plateau = sp >= 0.97 * vmax[:, None]
        self.nvalid = np.sum(np.isfinite(self.force_dc), axis=1)
        f = self.force_dc
        self.f_on = np.array([np.nanmedian(f[i, max(self.onset[i] - 3, 0):max(self.onset[i], 1)]) for i in range(n)])
        self.valid = np.isfinite(self.f_on) & (self.onset >= 2)
        pos0 = np.array([self.position[i, self.onset[i] - 1] if self.onset[i] > 0 else 0.0 for i in range(n)])
        self.x = self.position - pos0[:, None]
        self.df = f - self.f_on[:, None]
        # check column assignment against beta
        for s in ("v1", "v2", "v3"):
            m = self.folder == s
            mb = [np.median(self.beta[m & (self.col == k)]) for k in range(40)]
            assert np.all(np.diff(mb) > 0), (A, s)

    def at_travel(self, i, xg, need_moving=True):
        """first block index where travel >= xg while moving (or -1)."""
        ok = self.x[i] >= xg
        if need_moving:
            ok &= self.moving[i]
        j = np.flatnonzero(ok)
        return int(j[0]) if j.size else -1


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan
