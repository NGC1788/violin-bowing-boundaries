# Source future-path preparation gate

- This is one source-only data preparation audit, not model training or a performance experiment.
- Read only original train recordings and train labels. Do not open or hash old dev/test truth. Existing frozen artifacts remain unchanged.
- Hold a real planning lock and claim tasks before editing. Before execution obtain a separate code review, pass synthetic leakage tests, and freeze sources, allowed input hashes and contract.
- Run preparation once, within 600 seconds and the shared deadline. Preserve partial output and failures. Do not overwrite or refreeze after starting.
- Store prefix inputs separately from future-path labels. Never place actual future masks, lengths, targets or guards into inference inputs.
- Mask catalogues must be built from each inner train only. Unknown held patterns are reported, not added or dropped. All invalid records remain present.
- Expose only code, aggregate results and file hashes. Waveforms, identities, row-level paths, masks and targets stay private.
- A finite catalogue passing an operational coverage screen is not an observer model or an accuracy result. The conditional Gaussian assumptions and novelty remain unproven.
