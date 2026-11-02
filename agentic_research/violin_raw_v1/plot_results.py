"""Plot already-sealed aggregate results; no input/label/prediction arrays read."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    raw = (base / 'RESULTS.json').read_bytes()
    seal = json.loads((base / 'RESULTS_SEAL.json').read_text())
    if hashlib.sha256(raw).hexdigest() != seal['results_sha256']:
        raise ValueError('sealed results changed')
    result = json.loads(raw)
    methods = ['frequency', 'markov', 'hgb', 'mlp_ind', 'mlp_joint', 'mlp_offset']
    labels = ['Frequency', 'Markov', 'HGB', 'Independent\nMLP', 'Joint\nMLP', 'Offset\nMLP']
    fig, (ax, ci) = plt.subplots(1, 2, figsize=(11, 4.5),
        gridspec_kw={'width_ratios': [1.25, 1]}, layout='constrained')
    values = [result['test'][m]['log_loss'] for m in methods]
    bars = ax.bar(np.arange(6), values, width=.72,
        color=['#a7b2bf', '#27649b', '#a7b2bf', '#a7b2bf', '#a7b2bf', '#cf6a35'])
    ax.set_xticks(range(6), labels, fontsize=9); ax.set_ylim(0, .78)
    ax.set_ylabel('Participant-macro log loss (lower is better)')
    ax.set_title('Fixed twelve-trial test results', loc='left', fontweight='bold', fontsize=11)
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4, fontsize=9)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    for i, method in enumerate(['markov', 'mlp_joint', 'mlp_ind']):
        comparison = result['primary_offset_comparisons'][method]
        lo, hi = comparison['percentile_95_interval']; value = comparison['mean_difference']
        ci.errorbar(value, i, xerr=[[value - lo], [hi - value]], fmt='o',
                    color='#cf6a35' if i == 0 else '#52687e', capsize=4, ms=6)
    ci.set_yticks(range(3), ['vs Markov', 'vs Joint MLP', 'vs Independent MLP'])
    ci.set_ylim(2.6, -.6); ci.set_xlim(-.045, .21); ci.axvline(0, color='#666', ls='--', lw=1)
    ci.set_xlabel('Offset minus comparator log loss\n(positive = offset is worse)', fontsize=9)
    ci.set_title('Paired differences and exploratory 95% intervals', loc='left', fontweight='bold', fontsize=10)
    ci.grid(axis='x', alpha=.2)
    for axis in (ax, ci):
        for side in ['top', 'right']: axis.spines[side].set_visible(False)
    fig.suptitle('Violin forecast: the prespecified improvement screen failed', fontsize=13, fontweight='bold')
    fig.supxlabel('9 test participants / 389 recordings. MLPs: mean losses over 3 seeds. '
                  'Paired participant bootstrap: 5,000 resamples.', fontsize=8, color='#555')
    fig.savefig(base / 'RESULTS_FIGURE.png', dpi=220, facecolor='white')
    fig.savefig(base / 'RESULTS_FIGURE.svg', facecolor='white')
    svg = base / 'RESULTS_FIGURE.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__': main()
