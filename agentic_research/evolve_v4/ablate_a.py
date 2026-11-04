"""Ablate the a1 winner (t2) on the SEARCH split only: switch off one part at a time.

  .venv-mac/bin/python -m agentic_research.evolve_v4.ablate_a
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = (HERE / "runs/a1/programs/t2.py").read_text()

VARIANTS = {
    "full": [],
    "no_seed_opening (KSEED=0)": [("KSEED = 8", "KSEED = 0")],
    "longer_seed_opening (KSEED=16)": [("KSEED = 8", "KSEED = 16")],
    "no_rank0_shortcut": [("    if s0.any():", "    if False:")],
    "no_witness_verification (phase 1)": [("    ver.sort()", "    ver = []")],
    "no_bracket_refinement (phase 2)": [("    refs.sort()", "    refs = []")],
    "linear_scan_not_bisection": [("return col, int(ranks[np.abs(ranks - q).argmin()])", "return col, int(ranks[0])"),
                                  ("return c, int(ranks[len(ranks) // 2])", "return c, int(ranks[0])")],
    "no_pass_pinning (phase 3)": [("    # ---- 3. pass-style pinning", "    return _fallback(view)\n    # ---- 3. pass-style pinning")],
}

if __name__ == "__main__":
    out = {}
    folder = HERE / "runs/a1/ablation"
    folder.mkdir(exist_ok=True)
    for name, edits in VARIANTS.items():
        code = SOURCE
        for old, new in edits:
            if old not in code:
                raise ValueError(f"{name}: pattern not found: {old}")
            code = code.replace(old, new)
        path = folder / (name.split(" ")[0] + ".py")
        path.write_text(code)
        proc = subprocess.run([str(REPO / ".venv-mac/bin/python"), "-m", "agentic_research.evolve_v4.evaluate_a",
                               str(path), "search"], cwd=REPO, capture_output=True, text=True, timeout=900)
        m = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else None
        out[name] = dict(total=m["total"], totals=m["totals"], compatible=m["compatible_total"]) if m else proc.stderr[-300:]
        print(name, out[name], flush=True)
    (folder / "results.json").write_text(json.dumps(out, indent=1))
