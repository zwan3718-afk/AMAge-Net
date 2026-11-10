from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
RESULTS = Path(config.RESULTS_DIR)
O = Path(config.ANALYSIS_DIR) / 'publication_figures'
O.mkdir(parents=True, exist_ok=True)

models=['g','d','concat','cross','gate','amage'];labels=['G','D','Concat','Cross','Gate','Full']
sns.set_theme(style='white',font='DejaVu Sans')
plt.rcParams.update({'axes.titlesize':22,'axes.titleweight':'bold','axes.labelsize':19,'xtick.labelsize':15,'ytick.labelsize':17,'pdf.fonttype':42})
S={m:json.load(open(RESULTS/f'{m}_camcan/summary.json')) for m in models}
fig,axes=plt.subplots(1,4,figsize=(20,5.5))
for ax,(key,title) in zip(axes,[('mae','MAE'),('rmse','RMSE'),('r2','R²'),('pcc','PCC')]):
 mu=[S[m][key][0] for m in models];sd=[S[m][key][1] for m in models]
 bars=ax.bar(np.arange(6),mu,yerr=sd,capsize=4,color=sns.color_palette('deep',6),edgecolor='black',linewidth=1.5,error_kw={'elinewidth':1.1})
 top=max(np.array(mu)+sd)
 for b,v,e in zip(bars,mu,sd):ax.text(b.get_x()+b.get_width()/2,v+e+.018*top,f'{v:.2f}',ha='center',va='bottom',fontsize=17)
 ax.set_ylim(0,top*1.16);ax.set_xticks(range(6));ax.set_xticklabels(labels,rotation=35,ha='right')
 ax.set_title(title,pad=10);ax.set_ylabel(title,labelpad=4);sns.despine(ax=ax)
 for sp in ['left','bottom']:ax.spines[sp].set_linewidth(1.6)
 ax.tick_params(axis='both',pad=4)
fig.tight_layout(w_pad=1.5)
fig.savefig(O/'ablation-camcan.png',dpi=300,facecolor='white')
fig.savefig(O/'Fig10.pdf',facecolor='white')
print('Saved')
