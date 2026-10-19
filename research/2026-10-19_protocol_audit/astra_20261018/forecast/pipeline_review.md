# Pipeline durability and disk review — 2026-10-18

Read-only review of the live pipeline, archive companion, B1 recovery waiter, and published A1/A2 companion outputs. No process was started, stopped, restarted, or signalled; no running source was edited. Tests use temporary fixtures only.

## Verified snapshot

- Exactly one original Python `replicate.py`: PID 12819. Its `caffeinate` child is not a duplicate pipeline.
- Exactly one companion: PID 33702, scanning every 5 seconds. Exactly one B1 recovery waiter: PID 33979. Their `caffeinate` wrappers are also not duplicate workers.
- Original state marks A2, A1 and B1 feature extraction complete. Current child downloader PID 31675 is downloading C1 (`2024-04-12_TypeC_sample1.7z`). The last completed-chunk progress is 9.25 / 14.02 GiB; retry logs include premature EOF and resumable retries. No new failure should be inferred merely from those retry lines.
- C1's sparse `.part` file has **logical size 15,054,063,064 bytes but allocated size 10,293,608,448 bytes**. Logical size must never be interpreted as completed download progress or already consumed storage. The original downloader's chunk receipt is the progress authority.
- Free bytes at this snapshot: **116,748,603,392** (116.75 decimal GB; 108.73 GiB).
- Companion `pins/` and `staging/` are empty at this snapshot. The protected original A2 archive still exists.
- `test_stream_order.py` and all companion tests pass together: **18 passed in 0.11 s**. The stream-order fixture creates a real solid 7-Zip archive, compares each listed whole/beta member to its bytes in the filtered stream, and checks the total stream length. It demonstrates this ordering for the fixture and bundled executable; it is not a universal guarantee about every malformed archive.

## Published A1/A2 output verification

Both published output directories were reopened, independently validated against their complete feature-ID sets and original pipeline folder counts, and compared to their receipt's payload/coverage hashes. Both matched exactly. Each has 6,000 unique strokes and complete ramp/block beta agreement.

| Archive | Validated blocks | Receipt status | Raw integrity scope |
|---|---:|---|---|
| A1 | 450,631 | `legacy_outputs_validated` | No raw archive MD5 recheck; original was already deleted. Legacy extractor exit codes are unknown. |
| A2 | 457,271 | `complete` | Receipt records current protected raw archive size and published MD5 validation; legacy extractor exit codes remain unknown. |

A2's receipt has `commands: []` because existing outputs were validated and reused, not regenerated. Its recorded hashes for `ramp_meta.py`, `stroke_blocks.py` and the companion still match the current files. The internal A2 published receipt says `validated`, whereas the external control receipt says `complete`: that reflects publication preceding final pin release. These statuses should not be collapsed into a claim that the legacy extractor processes were observed to exit successfully.

## Priority finding: the 60 GB reserve is not guaranteed

The companion's `free >= 60 GB + 1 GB` check only bounds its own small outputs. It does **not** reserve space for the original runner's forthcoming folder extraction. The original runner checks free space only before downloading an archive. When the original runner deletes its archive, a failed or unfinished companion hardlink deliberately keeps those disk blocks allocated. One failed pin can therefore remove the headroom needed by the next extraction, even though the original path has disappeared.

The previous 24 GiB extraction allowance is not conservative. The root reviewer measured protected A2's selected whole+beta member sizes with the actual 7-Zip listing: **38.354 / 22.429 / 14.433 decimal GB** for its three folders. C1's exact folder sizes are not known until its download completes; A2 is only a planning proxy, not a verified bound for C1.

Using that proxy, current free space minus C1's ~4.76 GB remaining allocation, 38.354 GB extraction, and 1 GB outputs gives about **72.63 GB**. That projection has ~12.63 GB above the required reserve. One retained ~15 GB archive would exhaust that margin. Larger/longer future archives may need more than the A2 proxy. No indefinite unattended safety claim is justified by today's free-space snapshot.

### Minimal fail-closed guard

Before **every folder extraction**, use that completed archive's exact selected whole+beta member sizes from `7zz l -slt` and require:

`free_bytes >= 60_000_000_000 + selected_folder_bytes + bounded_output_allowance + other_current_writes_budget`.

Do not subtract the same pinned/original inode twice: its allocation is already reflected in `free_bytes`. Estimate remaining sparse download allocation from the declared bytes minus physically allocated bytes, with a conservative margin; logical `.part` size is unusable for this calculation. Better yet, schedule no new large download while extraction is active. Re-check after each completed folder, after download completion, and before starting recovery. On failure, preserve all files and explicitly stop scheduling new extraction/download work; do not delete failed pins automatically.

This cannot be enforced by the current passive companion alone. The least invasive controlled coordination opportunity is while the original parent is verified waiting on its download child: the root operator can hold the **parent** before that child exits, allow the download to finish, obtain the exact inventory, and resume the parent only if the next extraction is affordable. That is a proposal for a deliberate root action, not a change made by this audit. A race-prone watchdog that waits for free space to fall below 60 GB is too late. Any code-level boundary guard requires coordinating the running pipeline rather than editing the file and assuming a live Python process reloads it.

## Other durability findings

1. **Crash between directory publication and external receipt:** after `publish(stage, final)` but before the external `complete` receipt, a restart can see an existing valid final directory and re-run extraction, then fail because overwrite is refused. No output is overwritten and the pin is preserved, but progress requires manual recovery. Minimal repair: before launching a child, validate a pre-existing final directory against its internal receipt, exact inventory and raw checksum, then atomically promote the external receipt. Never blindly delete/rewrite the published directory.
2. **Unexpected basename collisions:** unchanged `stroke_blocks.py` stores only the leaf folder name. An archive containing `rep_1/v1` and `rep_2/v1` cannot safely be keyed by `(v1, trial)`. The companion rejects this before extraction and preserves the pin. Such a failure is expected to need an explicitly versioned parser that retains full identities; it must not be worked around by dropping duplicates. This is a concrete route by which pins may accumulate later.
3. **Output duration limit:** unchanged block extraction has a 160 × 50 ms = 8 s cap. The companion rejects records containing more than 160 complete blocks, preserving the archive. Extending the cap is a versioned extraction change, not a reason to waive completeness checks.
4. **Source provenance:** hashes are currently captured at archive-job start. A collaborator could edit the second extractor while the first is running, making the initial second-script hash stale. Capture and record each actual script hash immediately before launch and after completion; fail or mark provenance ambiguous if they differ. An immutable script snapshot would be stronger. Existing A1/A2 hashes presently match.
5. **Two extractors share parsing assumptions:** matching beta values between ramp and block files is necessary but not fully independent. For adopted legacy files, also joining each `(full_folder, trial)` beta to the already audited feature table would strengthen validation. Do not claim that passing the small stream-order fixture alone verifies all scientific features.
6. **B1 recovery waiter:** it properly waits for the original runner and downloader to exit and then checks reserve plus the full B1 archive and output allowance. It does not start a second replicate pipeline. It currently does not verify that the companion process is alive before its download or stop waiting if the companion disappears. Minimal repair: check companion lock/health before starting and periodically while waiting, and produce an explicit actionable status if it is absent. This should not trigger a duplicate companion automatically.
7. **Original feature writer remains nontransactional:** the legacy runner still records folder completion after flushing an open archive-wide gzip, and it can mark trials with error rows as processed. The companion protects ramp/block output integrity, not those feature files. Before scientific use, feature gzip integrity and exact trial/window coverage must be validated separately. This audit does not re-label legacy `done` states as fully validated scientific outputs.

## Suggested checks for a later coordinated patch

- Simulate final directory existing with `running` external receipt; recovery must validate and finish without invoking either extractor.
- Simulate one failed pin plus the next sparse download and a 38+ GB folder; guard must deny extraction while preserving all sources.
- Check inode-aware accounting, exact equality at the 60 GB threshold, missing inventory, changing disk usage, and interrupted download.
- Exercise multi-parent leaf collisions and recordings exceeding eight seconds; neither may be silently published.
- Change an extractor between stages; receipt must not claim the original hash was executed.
- Stop the companion in a fixture; B1 waiter must report its absence rather than falsely claim recovery completion.
