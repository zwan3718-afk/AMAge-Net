from pathlib import Path
import numpy as np,pandas as pd
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

models=['braingnn','gcn','bc_gcn','resnet','efficientnet','cnn_orig','mfformer','concat','ctransformer','setgfrn']
names=['BrainGNN','GCN','BC_GCN','ResNet','EfficientNet','3D-CNN','MFFormer','AMAge-Net-Concat','CTransformer','SET-GFRN']
sns.set_theme(style='white',font='DejaVu Sans')
plt.rcParams.update({'pdf.fonttype':42,'axes.titlesize':16,'axes.titleweight':'bold','axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12})
fig=plt.figure(figsize=(20,16))
for ds,color,lim,title,top,bottom,ty in [('camcan','#3A8DDE',(15,95),'A.  Cam-CAN',.945,.55,.985),('oasis','#3FB39A',(40,98),'B.  OASIS-3',.45,.055,.49)]:
 fig.text(.52,ty,title,ha='center',va='top',fontsize=24,fontweight='bold')
 gs=fig.add_gridspec(2,5,left=.052,right=.992,top=top,bottom=bottom,wspace=.32,hspace=.40)
 for i,(m,name) in enumerate(zip(models,names)):
  d=pd.concat([pd.read_csv(p) for p in sorted((RESULTS/f'{m}_{ds}').glob('fold*_test_predictions.csv'))])
  x,y=d.actual_age.values,d.predicted_age.values
  ax=fig.add_subplot(gs[i//5,i%5])
  sns.regplot(x=x,y=y,ax=ax,scatter_kws=dict(s=22,color=color,edgecolor='k',linewidths=.4,alpha=.75),line_kws=dict(color='k',lw=1.6),ci=95,seed=42)
  ax.plot(lim,lim,'k--',lw=1.1);ax.set_xlim(lim);ax.set_ylim(lim)
  ax.text(.05,.95,f'r = {np.corrcoef(x,y)[0,1]:.2f}\nMAE = {np.abs(y-x).mean():.2f}',transform=ax.transAxes,va='top',fontweight='bold',fontsize=14)
  ax.set_title(name,pad=8);ax.set_xlabel('True Age',labelpad=3);ax.set_ylabel('Predicted Age',labelpad=2)
  ax.set_xticks([20,40,60,80] if ds=='camcan' else [40,60,80])
  ax.set_yticks([20,40,60,80] if ds=='camcan' else [40,60,80])
  ax.grid(alpha=.4);sns.despine(ax=ax)
fig.savefig(O/'baseline.png',dpi=300,facecolor='white')
fig.savefig(O/'Fig5.pdf',facecolor='white')
print('Done')
