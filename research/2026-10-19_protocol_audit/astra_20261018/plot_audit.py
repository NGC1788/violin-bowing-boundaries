"""Recreate the audit figure from saved numeric results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).resolve().parent
r=json.loads((p/'results/history_prediction_audit.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
colors=['#277da8','#de754b']
for j,(tr,te) in enumerate([('A1','A2'),('A2','A1')]):
    vals=[next(f['metrics']['log_loss'] for f in r['fits'] if (f['train'],f['test'],f['learner'],f['inputs'])==(tr,te,'hist_gradient_boosting',k)) for k in ['instant','clock','history']]
    x=np.arange(3)+(j-.5)*.33
    axs[0].bar(x,vals,.31,label=f'{tr} train → {te} test',color=colors[j])
    for xx,v in zip(x,vals):axs[0].text(xx,v+.007,f'{v:.3f}',ha='center',fontsize=8)
axs[0].set_xticks(np.arange(3),['Current controls','+ time','+ full history'])
axs[0].set_ylabel('Log loss (lower is better)');axs[0].set_ylim(0,.46);axs[0].legend(fontsize=8)
axs[0].set_title('History concatenation failed to transfer',fontsize=11)
for j,tag in enumerate(['A1','A2']):
    rows=[z for z in r['inputs'][tag]['force_change_by_speed'] if z['onset_rank']==[1,20]]
    axs[1].plot([.05,.1,.2],[100*z['fraction_last_below_half_onset'] for z in rows],'o-',color=colors[j],label=tag)
axs[1].set_xticks([.05,.1,.2]);axs[1].set_xlabel('Bow speed (m/s)')
axs[1].set_ylabel('Strokes ending below half onset force (%)')
axs[1].set_ylim(-3,105);axs[1].legend()
axs[1].set_title('Lowest 20 onset-force ranks per column',fontsize=11)
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.suptitle('A1/A2 exploratory audit · waveform surrogate labels · no causal or equilibrium claim',fontsize=10)
fig.savefig(p/'results/history_audit.png',dpi=180);fig.savefig(p/'results/history_audit.pdf');plt.close(fig)
