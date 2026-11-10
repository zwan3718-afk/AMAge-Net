from pathlib import Path
import numpy as np
import pandas as pd
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

sns.set_theme(style='whitegrid',font='DejaVu Sans')
plt.rcParams.update({'font.size':14,'axes.labelsize':20,'axes.titlesize':23,'axes.titleweight':'bold','xtick.labelsize':16,'ytick.labelsize':16,'axes.edgecolor':'black','axes.linewidth':1.6,'grid.color':'.82','grid.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(14,6.6))
for ax,ds,title,color,lim in zip(axes,['camcan','oasis'],['Cam-CAN','OASIS-3'],['#1E90FF','#20B2AA'],[(15,93),(39,98)]):
 d=pd.concat([pd.read_csv(p) for p in sorted((RESULTS / f'amage_{ds}').glob('fold*_test_predictions.csv'))])
 x,y=d.actual_age.to_numpy(),d.predicted_age.to_numpy()
 sl,ic=np.polyfit(x,y,1);r=np.corrcoef(x,y)[0,1];mae=np.abs(y-x).mean()
 sns.regplot(x=x,y=y,ax=ax,ci=95,n_boot=1000,seed=42,color='black',scatter_kws={'s':30,'color':color,'edgecolor':'black','linewidths':.65,'alpha':.7},line_kws={'color':'black','lw':2.2})
 ax.plot(lim,lim,'--',color='.2',lw=1.8)
 ax.set(xlim=lim,ylim=lim,xlabel='True Age',ylabel='Predicted Age',title=title)
 ax.set_xticks([20,40,60,80] if ds=='camcan' else [40,60,80])
 ax.set_yticks([20,40,60,80] if ds=='camcan' else [40,50,60,70,80,90])
 ax.text(.05,.95,f'r = {r:.2f}\nMAE = {mae:.2f}',transform=ax.transAxes,va='top',fontsize=18,fontweight='bold',linespacing=1.55)
 ax.text(.05,.76,f'Slope = {sl:.2f}, intercept = {ic:.1f}',transform=ax.transAxes,va='top',fontsize=11.5)
 sns.despine(ax=ax);ax.tick_params(length=0)
fig.tight_layout(w_pad=2)
fig.savefig(O/'ours.png',dpi=600,facecolor='white')
fig.savefig(O/'Fig4.pdf',facecolor='white')
print('Saved Figure 4 PNG/PDF')
