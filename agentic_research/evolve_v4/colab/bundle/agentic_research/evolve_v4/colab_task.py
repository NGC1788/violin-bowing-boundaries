"""Search-split cases for the Colab bundle. The sealed C/D holdout is not shipped."""
import gzip
import hashlib
import json
import pickle
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "cases"


def load_cases():
    blob = (DATA / "cases_search.pkl.gz").read_bytes()
    expected = json.loads((DATA / "manifest.json").read_text())["cases_sha256"]
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("cases_search.pkl.gz hash mismatch")
    return pickle.loads(gzip.decompress(blob))
