#!/usr/bin/env python3
"""Plot every frozen A2 result without selecting a favorable source-speed pair."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
main=json.loads((HERE/'lockbox_A2/original_results.json').read_text())
transport=json.loads((HERE/'lockbox_A2/transport_results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(15,7.8),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
names=list(main)
scores=[main[n]['macro_panels']['family_min_iou'] for n in names]
colors=['#cd7221' if n=='covariant_logistic' else '#7153a1' if n=='covariant_family_robust' else '#8295a5' for n in names]
axes[0].barh(range(len(names)),scores,color=colors,height=.7)
axes[0].set_yticks(range(len(names)),[n.replace('covariant_','F/v: ').replace('rawlog_','Raw: ').replace('_',' ') for n in names],fontsize=9)
axes[0].invert_yaxis()
for i,s in enumerate(scores): axes[0].text(s+.01,i,f'{s:.4f}',va='center',fontsize=9)
axes[0].set_xlim(0,1)
axes[0].set_xlabel('Mean of three panel-wise minimum IoUs')
axes[0].set_title('Original protocol: all 14 models\nOne training speed: A / 0.10 m/s',fontsize=12)
methods=['affine_flow','translation_k0','learned_alpha','raw_3d']
labels=['Full affine flow','Matched k = 0','Learned speed exponent','Raw 3D quadratic']
palette=['#7153a1','#2890a2','#cd7221','#49545e']
keys=['A__0.05_0.1','A__0.05_0.2','A__0.1_0.2']
x=np.arange(3)
for j,(m,label,color) in enumerate(zip(methods,labels,palette)):
 vals=[transport[k]['primary_endpoint_by_model'][m] for k in keys]
 axes[1].bar(x+(j-1.5)*.19,vals,width=.18,label=label,color=color)
axes[1].set_xticks(x,['0.05, 0.10\nPRIMARY','0.05, 0.20\nSecondary','0.10, 0.20\nSecondary'])
axes[1].set_ylim(0,1)
axes[1].set_ylabel('Mean of three panel-wise minimum IoUs')
axes[1].set_xlabel('Frozen source-A training speed pair (m/s)')
axes[1].set_title('Separate transport addendum: all 12 models\nTwo training speeds per model',fontsize=12)
axes[1].legend(loc='upper center',frameon=False,fontsize=9)
for ax in axes:
 ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.18)
 ax.set_axisbelow(True)
fig.suptitle('New specimen A2: both prespecified primary comparisons fail',fontsize=16)
fig.supxlabel('5,892 eligible conditions on one specimen; fixed operational labels. No fitting on A2. Protocols have different training budgets.',fontsize=10)
fig.savefig(HERE/'lockbox_results.png',dpi=180)
