"""Data figures: Figure 4 (AMAge-Net predictions), Figure 5 (baselines), Figure 10 / S1 (ablation),
Figures S2, S3 (rank stability) and S4 (loss curves). Brain renderings (Figures 6-9) are made from
<ANALYSIS_DIR>/rendering/ with external software (e.g. BrainNet Viewer).

Usage: python analysis/figures.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

OUT = config.ANALYSIS_DIR
NAMES = {'braingnn': 'BrainGNN', 'gcn': 'GCN', 'bc_gcn': 'BC_GCN', 'resnet': 'ResNet', 'efficientnet': 'EfficientNet',
         'cnn_orig': '3D-CNN', 'mfformer': 'MFFormer', 'concat': 'AMAge-Net-Concat', 'ctransformer': 'CTransformer',
         'setgfrn': 'SET-GFRN', 'amage': 'AMAge-Net', 'g': 'AMAge-Net-G', 'd': 'AMAge-Net-D', 'cross': 'AMAge-Net-Cross',
         'gate': 'AMAge-Net-Gate'}
LIMS = {'camcan': (15, 95), 'oasis': (40, 98)}
COLORS = {'camcan': '#3A8DDE', 'oasis': '#3FB39A'}


def preds(tag):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(config.RESULTS_DIR, tag, 'fold*_test_predictions.csv'))])


def scatter(ax, df, color, title, lim, show_fit=False):
    x, y = df.actual_age.values, df.predicted_age.values
    sns.regplot(x=x, y=y, ax=ax, scatter_kws=dict(s=22, color=color, edgecolor='k', linewidths=0.4, alpha=0.75),
                line_kws=dict(color='k', lw=1.6), ci=95)
    ax.plot(lim, lim, 'k--', lw=1.1); ax.set_xlim(lim); ax.set_ylim(lim)
    sl, ic, r, *_ = stats.linregress(x, y)
    txt = f'r = {r:.2f}\nMAE = {np.abs(y - x).mean():.2f}' + (f'\nslope = {sl:.2f}, intercept = {ic:.1f}' if show_fit else '')
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold'); ax.set_xlabel('True Age'); ax.set_ylabel('Predicted Age')
    ax.grid(alpha=0.4); sns.despine(ax=ax)


def rank_plot(table, color, title, fname):
    t = pd.read_csv(table).sort_values('rank').head(23).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 7.5)); y = np.arange(len(t))
    ax.errorbar(t['rank'], y, xerr=[t['rank'] - t.rank_lo, t.rank_hi - t['rank']], fmt='o', color=color, ecolor=color, alpha=0.8, elinewidth=3)
    ax.set_yticks(y); ax.set_yticklabels(t.region, fontsize=8)
    ax.set_xlabel('Rank among 90 regions (point: full sample; bar: 95% bootstrap interval)'); ax.set_title(title)
    ax.grid(axis='x', alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=300); plt.close()


def main():
    sns.set_theme(style='white', font_scale=1.15)
    cohorts = [c for c in config.DATASETS if os.path.exists(os.path.join(config.RESULTS_DIR, f'amage_{c}', 'summary.json'))]
    # Figure 4
    fig, axs = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 5.5), squeeze=False)
    for ax, c in zip(axs[0], cohorts):
        scatter(ax, preds(f'amage_{c}'), COLORS[c], {'camcan': 'Cam-CAN', 'oasis': 'OASIS-3'}[c], LIMS[c], show_fit=True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig4_amagenet.png'), dpi=300); plt.close()
    # Figure 5
    order = ['braingnn', 'gcn', 'bc_gcn', 'resnet', 'efficientnet', 'cnn_orig', 'mfformer', 'concat', 'ctransformer', 'setgfrn']
    fig = plt.figure(figsize=(25, 10.5 * len(cohorts)))
    gs = fig.add_gridspec(2 * len(cohorts), 5, hspace=0.45, wspace=0.3)
    for blk, c in enumerate(cohorts):
        for i, m in enumerate(order):
            if os.path.exists(os.path.join(config.RESULTS_DIR, f'{m}_{c}')):
                scatter(fig.add_subplot(gs[blk * 2 + i // 5, i % 5]), preds(f'{m}_{c}'), COLORS[c], NAMES[m], LIMS[c])
    plt.savefig(os.path.join(OUT, 'fig5_baselines.png'), dpi=200); plt.close()
    # Figure 10 / S1
    abl = ['g', 'd', 'concat', 'cross', 'gate', 'amage']; pal = sns.color_palette('deep', len(abl))
    for c in cohorts:
        S = {m: json.load(open(os.path.join(config.RESULTS_DIR, f'{m}_{c}', 'summary.json'))) for m in abl}
        fig, axs = plt.subplots(1, 4, figsize=(24, 6.5))
        for ax, (k, lab) in zip(axs, [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²'), ('pcc', 'PCC')]):
            mu = [S[m][k][0] for m in abl]; sd = [S[m][k][1] for m in abl]; top = max(a + b for a, b in zip(mu, sd))
            bars = ax.bar(range(len(abl)), mu, yerr=sd, capsize=5, color=pal, edgecolor='k', linewidth=1.8)
            for b_, v, e in zip(bars, mu, sd):
                ax.text(b_.get_x() + b_.get_width() / 2, v + e + 0.01 * top, f'{v:.2f}', ha='center', va='bottom', fontsize=14)
            ax.set_ylim(0, top * 1.12); ax.set_xticks(range(len(abl))); ax.set_xticklabels([NAMES[m] for m in abl], rotation=40, ha='right')
            ax.set_title(lab, fontweight='bold'); ax.set_ylabel(lab); sns.despine(ax=ax)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, ('fig10_ablation_camcan.png' if c == 'camcan' else 'figS1_ablation_oasis.png')), dpi=250); plt.close()
    # Figures S2 / S3
    for mod, color, name in [('smri', '#C0392B', 'figS2_smri_rank_stability.png'), ('fmri', '#4C78A8', 'figS3_fmri_rank_stability.png')]:
        p = os.path.join(OUT, f'saliency_{mod}_table.csv')
        if os.path.exists(p):
            rank_plot(p, color, f'{mod[0]}MRI saliency rank stability (Cam-CAN, 1,000 resamples)', name)
    # Figure S4
    fig, axs = plt.subplots(1, len(cohorts), figsize=(5.5 * len(cohorts), 4.2), squeeze=False)
    for ax, c in zip(axs[0], cohorts):
        H = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(config.RESULTS_DIR, f'amage_{c}', 'fold*_history.csv')))]
        for col, lab, colr in [('train_loss', 'Training loss', '#4C78A8'), ('val_loss', 'Internal-validation loss', '#E45756')]:
            M = np.array([h[col].values for h in H]); e = np.arange(1, M.shape[1] + 1)
            ax.plot(e, M.mean(0), color=colr, label=lab); ax.fill_between(e, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=colr, alpha=0.2)
        ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss (normalized age)'); ax.set_title(c); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'figS4_loss_curves.png'), dpi=300); plt.close()
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
