# Portable retrospective forecasting benchmark

The package contains `dataset.npz` (about 8.6 MiB), its mandatory hash receipt `dataset.json`, `provenance.json`, and [data attribution](DATA_CREDITS.md). It contains no raw recordings and requires no download of the original large archives to run the prepared benchmark.

From the repository root, use the forecast-v3 loader:

```python
from pathlib import Path
from agentic_research.forecast_v3.data import load_prepared_dataset

dataset = load_prepared_dataset(
    Path('agentic_research/benchmarks/forecast_v3/dataset.npz')
)
assert dataset['X'].shape == (78000, 60)
train = dataset['split'] == 'train'
development = dataset['split'] == 'dev'
```

Outside the repository root, add the repository to `PYTHONPATH` and pass the absolute dataset path. The loader verifies the NPZ SHA256 against the adjacent receipt and loads with `allow_pickle=False`.

Array keys include `X`, `Y9`, `pattern_ids`, `prefix_Y9`, `prefix_pattern_ids`, `persistence_Y9`, `ids`, `sessions`, `groups`, `columns`, `measurement_sessions`, and `split`. `feature_names` and `metadata` describe the construction. `prefix_Y9` and `persistence_Y9` are identical labels from the latest permitted prefix; `Y9` is the future target. Never append targets or identifying strings to model inputs. Fit preprocessing using the training partition only.

The grouping field contains families A/B/C/D. Separate diagram IDs and acquisition folders are retained for descriptive analysis; they are not automatically independent specimens. Both development families were previously exposed. Details and verified observer parity appear in [the data audit](../../forecast_v3/DATA_AUDIT.md).

`SHA256SUMS` lists hashes for the package payload and documentation; as usual it does not hash itself. No predictive result is asserted by this package.
