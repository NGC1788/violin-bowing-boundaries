"""Scientific figures generated solely from the retained exact profiles."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Patch

HERE = Path(__file__).resolve().parent


def main():
    r = json.loads((HERE / 'results.json').read_text())
    targets = [('B_T2r1|', '/v2', 'B repeat 1, 0.10 m/s'), ('A4|', '/v3', 'A sample 4, 0.20 m/s')]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), layout='constrained')
    summaries = []
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dedede')
    for ax, (prefix, suffix, title) in zip(axes, targets):
        gi = next(i for i, g in enumerate(r['groups']) if g.startswith(prefix) and g.endswith(suffix))
        group = r['groups'][gi]
        with np.load(HERE / 'profiles' / f'{gi:02d}.npz') as z:
            u = np.r_[z['u_lo'][0], z['u_hi']]; v = np.r_[z['v_lo'][0], z['v_hi']]
            delta = np.maximum(0., z['gap'][:, :, 0] / 2)
            baseline = float(delta[np.searchsorted(z['u_hi'], .8), np.searchsorted(z['v_hi'], .5)])
            mesh = ax.pcolormesh(u, v, delta.T, cmap=cmap, norm=Normalize(0, .9), rasterized=True)
            if np.nanmin(delta) < .2 < np.nanmax(delta):
                ax.contour((u[:-1]+u[1:])/2, (v[:-1]+v[1:])/2, delta.T, levels=[.2], colors='white', linewidths=1)
        best = next(x for x in r['minimum_tolerance'] if x['group'] == group and x['p'] == -2 and x['policy'] == 'preserve_all')
        uncon = next(x for x in r['minimum_tolerance'] if x['group'] == group and x['p'] == -2 and x['policy'] == 'unrestricted')
        w = uncon['witness'] if prefix == 'B_T2r1|' else best['witness']
        ax.axvline(.8, color='black', alpha=.25, lw=.8); ax.axhline(.5, color='black', alpha=.25, lw=.8)
        ax.scatter([.8], [.5], c='white', edgecolors='black', s=55, marker='s', label='Original rule', zorder=5)
        ax.scatter([w['u']], [w['v']], c='#f44b42', edgecolors='white', s=65, label='Selected rule', zorder=6)
        value = uncon['minimum_epsilon'] if prefix == 'B_T2r1|' else best['minimum_epsilon']
        ax.set_title(title + f'\nselected tolerance = {value:.6f}')
        ax.set_xlabel('Periodicity floor u'); ax.set_ylabel('Second-flyback-ratio ceiling v')
        ax.set_xlim(.7, .9); ax.set_ylim(.3, .7)
        ax.text(.98, .97, f"{w['all_edits']:,} label changes across all data\n{len(w['invalid_groups'])} folders with lost boundaries", transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox={'facecolor':'white', 'alpha':.92, 'edgecolor':'none'})
        summaries.append({'group': group, 'baseline_epsilon': baseline, 'best_epsilon': uncon['minimum_epsilon'],
                          'best_preserve_all_epsilon': best['minimum_epsilon'], 'selected': {'u':w['u'],'v':w['v'],'all_edits':w['all_edits'],'invalid_groups':len(w['invalid_groups'])}})
    axes[0].legend(loc='center right', fontsize=8, framealpha=.9)
    fig.colorbar(mesh, ax=list(axes), label='Minimum required log-force tolerance (p = -2)', shrink=.88, extend='max')
    fig.suptitle('Exact joint-rule cells: local law tolerance, with global consequences at selected rules\nGrey: target boundary unresolved. White contour: tolerance 0.2. Cells are not probabilities.', fontsize=11)
    fig.savefig(HERE / 'fig_joint_regions.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 4.6), layout='constrained')
    values = [summaries[1]['baseline_epsilon'], summaries[1]['best_preserve_all_epsilon']]
    labels = ['Original rule\n0 edits', 'Best shared pair\nAll boundaries retained\n150 edits / 78,000 windows']
    bars = ax.barh([1, 0], values, color=['#9b9b9b', '#267aab'], height=.5)
    ax.axvline(.2, color='#bd4138', ls='--', label='Primary tolerance 0.2')
    for bar, value in zip(bars, values): ax.text(value+.006, bar.get_y()+bar.get_height()/2, f'≈ {value:.9f}', va='center', fontsize=11)
    ax.set_yticks([1,0], labels); ax.set_xlim(0, .49); ax.set_xlabel('Required log-force tolerance for A4, 0.20 m/s (p = -2)')
    fig.suptitle('The operational tolerance controls the no-loss conclusion\nCompatibility begins near 0.205553; the chosen cutoff was 0.2', fontsize=13)
    ax.legend(loc='upper right'); ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE / 'fig_tolerance_cutoff.png', dpi=160); plt.close(fig)
    certificates = json.loads((HERE / 'theory/obstruction_certificates.json').read_text())['triple_certificates']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6), layout='constrained')
    compact = []
    colors = ['#64a5cf', '#f2ad67', '#91c888']
    for ax, cert in zip(axes, certificates):
        chosen = cert['chosen_triple']; ids = chosen['ids']; betas = chosen['beta']
        pairs = sorted({tuple(sorted(ids.index(i) for i in rect['witness'])) for rect in cert['rectangles']})
        color = {pair:colors[i] for i,pair in enumerate(pairs)}
        for rect in cert['rectangles']:
            assert rect['reason'] == 'separated'
            pair = tuple(sorted(ids.index(i) for i in rect['witness']))
            u, v = rect['u'], rect['v']
            ax.add_patch(Rectangle((u['low'],v['low']),u['high']-u['low'],v['high']-v['low'],
                                   facecolor=color[pair],edgecolor='white',linewidth=.35))
        ax.scatter([.8],[.5],c='white',edgecolors='black',s=55,zorder=5)
        ax.set_xlim(.7,.9);ax.set_ylim(.3,.7)
        ax.set_xlabel('Periodicity floor u');ax.set_ylabel('Second-flyback-ratio ceiling v')
        name = 'A1, 0.20 m/s' if cert['group'].startswith('A1|') else 'B repeat 1, 0.05 m/s'
        ax.set_title(f"{name}, p = {cert['p']:g}\nThe three selected boundaries remain finite throughout")
        handles = [Patch(facecolor=color[pair],label=f'Conflict: β={betas[pair[0]]:.4f} and {betas[pair[1]]:.4f}') for pair in pairs]
        ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=9,frameon=False)
        compact.append({'group':cert['group'],'p':cert['p'],'beta':betas,'minimum_columns':3,'rectangles':cert['n_rectangles']})
    fig.suptitle('A changing conflicting pair can require three fixed columns over the full rule domain\nEvery colored region is incompatible at tolerance 0.2. No fixed pair alone covers the whole domain.',fontsize=12)
    fig.savefig(HERE / 'fig_fixed_obstructions.png',dpi=160);plt.close(fig)
    (HERE / 'figure_data.json').write_text(json.dumps({'targets':summaries,'primary_epsilon':.2,'triple_certificates':compact}, indent=2)+'\n')
    print('Wrote joint-cell, tolerance-cutoff and minimum-obstruction figures.')


if __name__ == '__main__': main()
