"""Size of the two search splits (search only; the sealed holdouts are not read).

Reads only `colab/bundle/cases/cases_search.pkl.gz` (violin search, 54 cases) and
`colab/bundle_robust/cases/cases_search.pkl.gz` (AI-audit search, 40 cases), checks
their sha256 against each bundle's manifest, and prints the number of cases,
columns per case and the number of waveform cells (= reads needed to read everything).

    python agentic_research/evolve_v4/figs/split_size.py
"""
import gzip
import hashlib
import json
import pickle
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLES = {
    "violin search": HERE.parent / "colab" / "bundle" / "cases",
    "AI-audit search": HERE.parent / "colab" / "bundle_robust" / "cases",
}


def load(cases_dir):
    blob = (cases_dir / "cases_search.pkl.gz").read_bytes()
    expected = json.loads((cases_dir / "manifest.json").read_text())["cases_sha256"]
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"hash mismatch in {cases_dir.name}")
    return pickle.loads(gzip.decompress(blob))


def main():
    for name, cases_dir in BUNDLES.items():
        cases = load(cases_dir)
        shapes = [c["labels"].shape for c in cases]  # (columns, rules, ranks)
        cols = [s[0] for s in shapes]
        cells = sum(s[0] * s[2] for s in shapes)
        groups = len({c["group"] for c in cases})
        compatible = sum(bool(c["expected"]) for c in cases)
        print(
            f"{name}: cases {len(cases)} (groups {groups} x 2 eps), "
            f"compatible {compatible}, rules {shapes[0][1]}, ranks {shapes[0][2]}, "
            f"columns min/median/max {min(cols)}/{statistics.median(cols):g}/{max(cols)}, "
            f"cells (read-everything) {cells}"
        )


if __name__ == "__main__":
    main()
