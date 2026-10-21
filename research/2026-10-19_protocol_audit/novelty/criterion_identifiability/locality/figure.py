from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
d=json.loads((P/'results.json').read_text())
names=list(d['summary']);labels=['Amplitude','Flyback height','Periodicity','2nd-flyback ratio']
y=np.arange(4);fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for key,delta,color,label in [('lower_endpoint_changed',-.22,'#1565a8','Lower endpoint changed'),('upper_endpoint_changed',0,'#d45a3b','Upper endpoint changed'),('band_lost',.22,'#8c9095','No K=2 run remains')]:
 val=np.array([d['summary'][n][key]*100 for n in names]);ax.barh(y+delta,val,height=.21,color=color,label=label)
 for i,v in enumerate(val):ax.text(v+.5,i+delta,f'{v:.1f}%',va='center',fontsize=9)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,56);ax.set_xlabel('Fraction of initially banded columns (%)')
ax.set_title('Equal trimming fraction, different boundary exposure',loc='left',fontweight='bold')
ax.legend(loc='lower right',fontsize=9);ax.spines[['top','right']].set_visible(False)
fig.text(.04,-.045,'13 sessions; all original tilt values; 15% trimming of eligible H labels.\nColumns recur across tilts: descriptive fractions, not independent samples.',fontsize=9)
fig.savefig(P/'fig_endpoint_locality.png',dpi=190,bbox_inches='tight')
fig.savefig(P/'fig_endpoint_locality.svg',bbox_inches='tight');plt.close(fig)
