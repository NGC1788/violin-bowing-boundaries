# Modern tabular comparator and metadata-only holdout inventory

Checked 2026-10-17. No checkpoint, waveform, or unused target label was downloaded or inspected for this note. The subsequently discovered `violin-rne/timeres` pipeline takes precedence over this inventory: local absence alone is not proof that a specimen has never been inspected.

## Comparator decision

**TabICLv2 is the first practical modern comparator.** Use the published 2.2.0 package, a fixed checkpoint, and exactly the same source rows, feature choices, and grouped validation as the candidate. It is a baseline, not a novel method. PyPI metadata reports a 310,448-byte wheel, Python ≥3.10, and required torch, sklearn, numpy, scipy, einops, psutil, tqdm, and huggingface-hub. Current Git main additionally lists tokenizers; pin the release rather than silently following main. Existing Python 3.11 / torch 2.14.1 meets the principal requirements. [Package metadata](https://pypi.org/pypi/tabicl/2.2.0/json), [official dependencies](https://github.com/soda-inria/tabicl/blob/main/pyproject.toml).

The official classifier checkpoint is `tabicl-classifier-v2-20260212.ckpt`, 110,368,038 bytes, SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`; Hugging Face revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, ungated, BSD-3-Clause. Pin both revision and hash. [Checkpoint metadata](https://huggingface.co/jingang/TabICL/blob/main/tabicl-classifier-v2-20260212.ckpt), [model card](https://huggingface.co/jingang/TabICL).

Official documentation supports CPU/MPS, an eight-member ensemble and configurable ensemble batch size. Training sets of 2k–4k rows with three features fall within the documented pretraining size range. Start with `device='cpu', n_estimators=8, batch_size=1, kv_cache=False, n_jobs=4` in an isolated environment; benchmark one source task before a sweep. M2 runtime and peak memory are **not measured**, and GPU timing in the paper is not an M2 estimate. MPS can be separately checked for numerical consistency. [Official implementation](https://github.com/soda-inria/tabicl), [2026 paper](https://arxiv.org/abs/2602.11139).

**TabPFN-3.5, not 2.5, is the current alternative.** The September 2026 model has local inference and a 5k-row CPU guard; current PyPI package is 9.2.0. Its code and weights have separate licenses; weight access requires a Prior Labs login/license acceptance. No authentication or acceptance was attempted. Dependency footprint is larger (including lightgbm, skrub and Apple-arm64 mlx). This is a useful optional second comparator after access is available, not a reason to delay the ungated baseline. [Official repo](https://github.com/PriorLabs/TabPFN), [model versions/access](https://docs.priorlabs.ai/models), [package metadata](https://pypi.org/pypi/tabpfn/9.2.0/json).

## Additional specimens: metadata only

All eight Zenodo record APIs were read. They describe cello G2 monochord experiments, not whole violins or independent laboratories. Metadata identify these plausible **future frozen** tests:

| Archive | Record | Compressed bytes | Scientific role |
|---|---:|---:|---|
| 2024-04-22_TypeA_sample3.7z | 17782542 | 14,416,522,615 | Same-type specimen replication |
| 2024-04-24_TypeA_sample4.7z | 17782552 | 14,688,771,984 | Another same-type specimen |
| 2024-04-26_TypeD_sample2.7z | 17782552 | 14,388,913,034 | Replication in another construction type |
| 2024-06-19_TypeA_T2_sample1.7z | 17782565 | 14,978,195,174 | Tension change, not automatically independent specimen |
| 2024-07-03_TypeB_T2_sample2.7z | 17782538 | 13,566,750,802 | Additional specimen at higher tension |
| 2024-10-23_variability_setup.7z | 17795326 | 28,218,620,427 | Setup-repeat sensitivity, not independent laboratory |

Sources: [Part 2](https://zenodo.org/api/records/17782542), [Part 3](https://zenodo.org/api/records/17782552), [Part 4](https://zenodo.org/api/records/17782565), [Part 5](https://zenodo.org/api/records/17782538), [Part 6](https://zenodo.org/api/records/17795326). All specify CC BY 4.0. Public metadata describe cold/warm as the **same string** before/after play-in. Parts 7–8 use 4–12 N and reduced β range: useful stress tests, not a replacement for normal-range replication.

A2 is exposed. A1/B1/C1/D1 were already processed locally. Before assigning an untouched holdout, reconcile the active `timeres` state and all prior analyses. Freeze the method, preprocessing and endpoint before examining that archive's target outcomes. Prefer several specimen-level tests; thousands of correlated grid cells do not create thousands of independent specimens. This note deliberately does not initiate downloads or modify the running pipeline.
