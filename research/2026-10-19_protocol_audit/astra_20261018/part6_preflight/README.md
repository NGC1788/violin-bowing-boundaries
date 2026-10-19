> **2026-10-19: 이 문서는 과거 상태 기록이다. disk_gate.py는 실행하지 않았고, 저장 PID가 오래되어 현재 실행하면 안 된다. 다운로드는 사용자 요청으로 중지되어 있다. 최신 상태는 ../RESEARCH_STATUS.md가 아닌 ../../astra_20261019/RESEARCH_STATUS_20261019.md 및 COLLAB_LOG.md를 따른다.**

# Part 6 preflight: format, disk and experimental design

Status at **2026-10-18 13:07 KST**. Metadata-only review; no Part 6, B1 or C1 outcome was opened, labelled, fitted, or scored. The archive was still incomplete, so **no archive listing was attempted**. Claude owns the live `replicate.py`, its restart procedures, and `session_variability.py`; none was edited or signalled.

## What the verified metadata actually establish

The local Zenodo metadata snapshots fetched at 02:12 and 03:17 UTC agree:

| Field | Value |
|---|---|
| Record | `17795326` |
| Archive | `2024-10-23_variability_setup.7z` |
| Published compressed bytes | `28,218,620,427` = 26.280 GiB |
| Published checksum | `md5:bddef0d9cf48a6dccbc27021116274a5` |
| Description | Setup variability, two repeated Schelleng diagrams under the same conditions |
| Collection | Cello G2 string on a robotic monochord, sampled at 50 kHz |

Source: [Zenodo Part 6 API](https://zenodo.org/api/records/17795326), locally preserved in `data/manifests/zenodo_17795326_20261018T031735.564420Z.json` and the earlier snapshot. The general description names three bow speeds and four channels, but does not establish this archive's actual directory structure, exact stroke count, exact string specimen, session randomization, remounting procedure, elapsed time between repeats, or rosin/bow state.

Its generic narrative describes indexed directories with `whole.csv`, while existing processed archives used `whole_<n>.csv` under speed folders. Treat that disagreement as unresolved metadata, not evidence that either parser already supports Part 6. Likewise, the narrative's numbered-folder speed assignment must not override the measured speed channel: previous archives already demonstrated inconsistent folder naming.

## Live state and immediate disk concern

One original pipeline is running: PID **41262**, with downloader PID **41266** and `--connections 16`. Companion PID **33702** and B1 recovery waiter PID **33979** remain separate single instances. Their `caffeinate` processes are wrappers, not duplicate pipelines. A Claude-owned waiter will call the session-analysis script when the original state becomes done/error; this review did not execute it.

The chunk receipt had 201 completed 64 MiB chunks, totaling **13,488,881,664 bytes**. Remaining declared download was **14,729,738,763 bytes**. The sparse partial file's logical length was already the full 28.219 GB, whereas physical allocation was **14,034,124,800 bytes**. Free disk was **109,651,451,904 bytes** (109.65 decimal GB). There were no companion hardlinks retained at this snapshot. Values naturally change as downloading continues.

**The 60 GB reserve is not currently guaranteed for the next extraction.** After allocating the remaining ~14.18 GB of the archive, about 95.47 GB would remain, before any output. The root reviewer measured a **38.354 GB** slow-speed whole+beta folder in protected A2. If Part 6 has a similarly sized folder, extracting it plus 1 GB of output allowance would leave approximately **56.11 GB**, below the reserve. This is a planning warning, not a claim that Part 6 has that size. Part 6's exact uncompressed folder sizes are unknown until the completed, checksum-verified archive can be listed.

Before any extraction, the pipeline owner should coordinate a boundary at which to list the verified completed archive and enforce:

`free_bytes >= 60_000_000_000 + exact_selected_folder_bytes + output_allowance + concurrent_write_budget`.

The current original runner checks only the pre-download free-space threshold. The companion streams into compact outputs, but its own output-budget check cannot constrain the original runner's larger extraction. Editing a running Python file does not install a guard in its already-loaded code. Do not delete protected A2, cache data, failed pins, or incomplete outputs to satisfy the inequality. No process action was taken by this reviewer; the root was notified of the projection for coordination with Claude.

## Compatibility checklist after checksum verification

These are structural checks only and can precede outcome inspection.

1. Verify exact compressed bytes and full published MD5, then obtain a 7-Zip member listing. Preserve a listing hash and a readable inventory of full paths, member sizes and diagram/repetition identifiers.
2. Establish the actual hierarchy and trial identity. Retain `(archive, full diagram/repetition path, full speed folder, trial)`. If trial numbers restart across repeats, they are not globally unique. Do not assume 12,000 strokes merely by multiplying two diagrams by the previous 6,000-stroke layout.
3. Require one matching whole signal and beta value per trial and inventory counts per full folder. Check duplicate/missing identities and all selected byte totals. A zero-member inventory must remain an error with archive retained, not an empty successful dataset.
4. **Known companion limitation:** unchanged `stroke_blocks.py` reduces a folder to its leaf name. If Part 6 contains `repeat_1/v1` and `repeat_2/v1`, identical trial numbers collide. The current companion deliberately rejects ambiguous leaves and keeps its hardlink. A versioned adapter retaining full paths would be required; ignoring the collision or deduplicating rows would mix experiments.
5. A literal `<trial>/<speed>/whole.csv` hierarchy also requires a versioned adapter. Do not broaden a regex until member pairs, stream order, naming and output identities have an executable fixture.
6. Check signal duration against `stroke_blocks.py`'s fixed 160 × 50 ms = 8 s cap. The companion rejects recordings with more than 160 complete blocks. Increasing the cap must be recorded as an extraction-version change.
7. Validate the feature gzip, exact trial/window coverage and error rows independently of legacy `status='done'`. The original feature writer is still archive-wide gzip with nontransactional folder updates. Validate companion ramp/block outputs, beta agreement and receipt hashes before joining them.

## What two repeated diagrams can answer

The useful immediate target is a **paired repeat difference under the observed protocol**: for the same nominal speed and compatible measured force/position/history, how much do boundary level, observable oscillation probability and measured control history differ between the two observed repetitions? Keep each repeat's measured controls and censoring explicit. This can reveal whether a claimed speed effect is smaller than a particular repeat-to-repeat discrepancy. It cannot automatically attribute that discrepancy to remounting, drift, rosin, sensor calibration, or the specimen.

Two repeats do **not** establish a reliable population distribution of session effects. At a fixed speed they supply only one between-repeat contrast. Under a strong normal random-effects assumption, a variance estimate can be computed with effectively one between-repeat degree of freedom, but its uncertainty is extremely large and the distributional assumption cannot be checked. In a distribution-free sense, two observed sessions cannot characterize the population session variance. Hundreds of beta-force cells and thousands of overlapping windows do not increase the number of independent sessions.

A within-grid or stroke bootstrap quantifies conditional uncertainty for those recorded diagrams; it is not a confidence interval for population session variance. A random-effects package producing a precise-looking standard error does not overcome this design. Report the two observed levels, their paired contrast, and sensitivity to control matching. Avoid claims such as “setup variance is small in general” or “speed has been deconfounded” from this comparison alone.

If both diagrams contain all three speeds, the repeats provide a blocked comparison only under additional assumptions about a common diagram effect and comparable recording order. Separate speed folders can still have their own setup state, and systematic speed-specific calibration or order effects remain inseparable from speed. Metadata alone do not establish randomized order, a crossover design, or independent session sampling. Repetition of the same lab setup also does not provide external laboratory or whole-violin validation.

This preflight fixes no scientific result. Part 6 outcomes remain assigned to the coordinated session analysis; B1/C1 remain unscored by this reviewer. No GitHub push was performed.

## Standalone boundary gate prepared for root review

`disk_gate.py` is a separate, fail-closed Part 6 gate; it does not edit or launch `replicate.py`. It is **not launched by this reviewer**. On root-authorized execution it verifies exact Python command, parent-child relation, Part 6 destination and process birth times, then holds only the original parent while its existing downloader continues. It verifies the completed raw checksum and actual inventory and resumes only when compatibility and `60 GB + largest selected folder + 2 GB` checks pass. It never resumes a parent that was already stopped before the gate began. Cancellation, timeout, format mismatch and insufficient space all retain the stopped parent with an explicit receipt. A zombie downloader is identified by PID and birth time because macOS may replace its command string after exit.

Tests: **10 passed** using mocked signals/processes; actual running parent/downloader identities were checked read-only. Neither test nor identity check sent a real signal. Root must coordinate any eventual launch with Claude. This guards Part 6's extraction boundary only; later archive scheduling still needs its own budget checks.
