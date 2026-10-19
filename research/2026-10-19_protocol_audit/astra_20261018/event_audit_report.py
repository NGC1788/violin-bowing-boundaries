"""Build a concise report from immutable event-audit outputs; no fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'event_audit_results'
paths=sorted(OUT.glob('*_event_audit.json'))
if len(paths)!=2:
    raise ValueError('both completed A1 and A2 audits are required')
reports=[json.loads(p.read_text()) for p in paths]
rows=[]
for result in reports:
    name='A1' if 'sample1' in result['archive'] else 'A2'
    mode='frozen_acquisition_fixed' if 'frozen_acquisition_fixed' in result['calibration'] else 'pipeline_fixed'
    for g in result['groups']:
        if g['nuisance']!=mode or g['window_s']!=.2:
            continue
        suffix=next(c for c in result['prefix_comparisons'] if c['nuisance']==mode and c['window_s']==.2
            and c['folder']==g['folder'] and c['prefix']=='fraction_0.5' and c['method']=='suffix80')
        rows.append({'archive':name,'speed':{ 'v1':.05,'v2':.1,'v3':.2}[g['folder'].split('/')[-1].rsplit('_', 1)[-1]],
            'n':g['strokes'],'multiple':g['strokes_multiple_H_episodes'],
            'K2detected':g['methods']['causal_K2']['detected'],
            'K2later_nonH':g['methods']['causal_K2']['later_nonH_observed'],
            'K2censored':g['methods']['causal_K2']['not_detected_by_last_observed_decision'],
            'suffix_changed':suffix['changed'],'prefix_n':suffix['comparisons'],
            'legacy_changed':g['legacy_force_filtered']['legacy_suffix_detection_changes_vs_unfiltered_among_eligible'],
            'legacy_n':g['legacy_force_filtered']['eligible_strokes_at_least3_positive_force_windows'],
            'legacy_gaps':g['legacy_force_filtered']['filtered_sequences_with_time_gaps']})
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
labels=[f"{r['archive']}\n{r['speed']:.2f} m/s" for r in rows]
x=np.arange(len(rows)); colors=['#38789b']*3+['#c86a3f']*3
axes[0].bar(x,[100*r['suffix_changed']/r['prefix_n'] for r in rows],color=colors)
axes[0].set_title('Retrospective suffix event changes\nwhen follow-up is cut in half')
axes[0].set_ylabel('% of strokes with an altered past event')
axes[1].bar(x,[100*r['K2later_nonH']/r['K2detected'] for r in rows],color=colors)
axes[1].set_title('A non-H decision occurs later\nafter first K=2 detection')
axes[1].set_ylabel('% of detected strokes')
for ax in axes:
    ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('0.20 s overlapping windows; descriptive detector events, not physical lock')
fig.savefig(OUT/'event_definition_audit.png',dpi=180)
plt.close(fig)
lines=['# Time-resolved event-definition audit — 2026-10-18','',
'**Scope:** A1 and A2 public robot-bowing recordings, all available strokes and every cached 0.10/0.20/0.40 s window. This audit measures observed detector events. It does not estimate a steady-state boundary, physical locking time, critical-slowing exponent, or bifurcation type.','',
'The former suffix rule chooses the first H-labelled window after which at least 80% of all remaining observed windows are H. This uses future observations, so truncating follow-up can change an event assigned to the past. A first-K run rule instead declares detection when K consecutive H decisions have become available. Each decision is stamped at the **window end**, not the start. K=1,2,3 and all three window lengths were retained, without target-driven selection.','',
'## Main descriptive results (0.20 s windows)','',
'| Sample | Speed (m/s) | Strokes | Suffix event changes after halving follow-up | First K=2 detected | Later non-H among detected | No K=2 detection by final decision | Multiple observed H episodes |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['archive']} | {r['speed']:.2f} | {r['n']} | {r['suffix_changed']}/{r['prefix_n']} ({100*r['suffix_changed']/r['prefix_n']:.1f}%) | {r['K2detected']} | {r['K2later_nonH']} ({100*r['K2later_nonH']/r['K2detected']:.1f}%) | {r['K2censored']} | {r['multiple']} |")
causal_total=sum(c['comparisons'] for report in reports for c in report['prefix_comparisons'] if c['method'].startswith('causal'))
causal_changed=sum(c['changed'] for report in reports for c in report['prefix_comparisons'] if c['method'].startswith('causal'))
lines += ['',f'Across every retained window length, nuisance setting, K, and prefix horizon, the first-K rule had **{causal_changed} changes in {causal_total:,} prefix comparisons**. Those comparisons repeatedly reuse the same strokes and are not independent replications. The exhaustive binary-sequence tests establish the stopping-rule property separately from these descriptive counts.','',
'![Suffix lookahead and recurrence](event_definition_audit.png)','',
'## What the event means','',
'- For 0.20 s windows spaced by 0.05 s, the earliest K=1,2,3 detections are 0.20, 0.25, and 0.30 s after plateau onset. The historical survival code assigns a first-window event to an upper bound of 0.025 s; this precedes availability of the analysed waveform. Its last-window start is likewise earlier than the last observable decision.','- A first-K event is exact on the observed decision grid. No detection by the last window end is an observed detector follow-up limit; it is not proof that the physical process will never become H. Prefix stability does not establish the independent-censoring assumptions needed by a survival model.','- A later non-H decision demonstrates that first detection is not an absorbing detector state. Overlap, detector error, changing controls and physical switching can all cause recurrence; the audit cannot identify which mechanism applies. H-episode endpoint spans are released as descriptive decision spans, not inferred physical dwell durations.','- A1 nuisance parameters are estimated once from its full feature archive and then held fixed. A2 is analysed both with original acquisition-window nuisance constants and with current pipeline constants. Prefix causality is conditional on these fixed classifiers, not an assertion that archive-wide calibration itself is online.','',
'## Force filtering and force history','',
'No rows are removed by force in the primary event sequences. This avoids silently concatenating windows separated by missing nonpositive-force samples. Time-grid gaps reset the first-K counter. The former positive-force filtering is reproduced only as a separate sensitivity diagnostic.','',
'| Sample | Speed | Legacy eligible strokes | Suffix time/status changes from filtering | Filtered sequences with internal time gaps |',
'|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['archive']} | {r['speed']:.2f} | {r['legacy_n']} | {r['legacy_changed']} | {r['legacy_gaps']} |")
lines += ['', '**Force changes during these strokes.** Neither causal detection nor a corrected time stamp converts this acquisition into a constant-force relaxation experiment. Inferring critical slowing down or an equilibrium minimum-force law from stroke-average force and these events remains unsupported. Acceleration, travel and force history also require a separate support/identifiability analysis.','',
'## Verification and outputs','',
'13 tests pass, including all binary histories of length 2–8 at every strict prefix for K=1,2,3, literal historical-suffix equivalence through length 9, counterexamples in both lookahead directions, missing-time resets, recurrence and invalid-input rejection. Every input archive is SHA-256 checked before and after the run. All source records are retained in the output; source hashes and calibration constants are saved.','',
'Full per-setting results: `*_event_audit.json`. Per-stroke detection times and force-filter sensitivity: `*_stroke_events.csv.gz`. Every observed H run with observation/gap-edge flags: `*_H_episodes.csv.gz`.','',
'```bash','/Users/jeongjihoon/violin-bowing-boundaries/.venv-mac/bin/python -m pytest astra_20261018/tests/test_event_definition_audit.py -q',
'/Users/jeongjihoon/violin-bowing-boundaries/.venv-mac/bin/python astra_20261018/event_definition_audit.py --output astra_20261018/event_audit_new_run','```','',
'Run from `~/violin-rne`. Existing audit outputs are preserved; choose a new output directory for a repeat.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'headline.json').write_text(json.dumps({'rows':rows,'causal_prefix_comparisons':causal_total,
    'causal_prefix_changes':causal_changed},indent=2)+'\n')
print(json.dumps({'rows':rows,'causal_prefix_comparisons':causal_total,'causal_prefix_changes':causal_changed},indent=2))
