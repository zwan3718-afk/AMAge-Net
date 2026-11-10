"""Paired comparisons of every method with AMAge-Net (Table 1, Table S2) and of the ablation variants with the full
model (Table 3, Table S8): subject-level sign-flip permutation test on absolute errors, bootstrap CI of the paired
MAE difference, Benjamini-Hochberg FDR over the ten baseline comparisons within each cohort.

Usage: python analysis/compare.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

BASELINES = ['braingnn', 'gcn', 'bc_gcn', 'resnet', 'efficientnet', 'cnn_orig', 'mfformer', 'ctransformer', 'concat', 'setgfrn']
ABLATION = ['g', 'd', 'concat', 'cross', 'gate']
rng = np.random.default_rng(0)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for r, i in list(enumerate(o))[::-1]:
        prev = min(prev, p[i] * n / (r + 1)); q[i] = prev
    return q


def preds(tag):
    files = glob.glob(os.path.join(config.RESULTS_DIR, tag, 'fold*_test_predictions.csv'))
    return pd.concat([pd.read_csv(f) for f in files]).set_index('subject_id').sort_index()


def paired(diff, n_perm=10000, n_boot=10000):
    perm = (rng.choice([-1, 1], size=(n_perm, len(diff))) * diff).mean(1)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return dict(dMAE=diff.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                p=(np.sum(np.abs(perm) >= abs(diff.mean())) + 1) / (n_perm + 1))


def main():
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    rows = []
    for cohort in config.DATASETS:
        if not os.path.exists(os.path.join(config.RESULTS_DIR, f'amage_{cohort}', 'summary.json')):
            continue
        ref = preds(f'amage_{cohort}'); e_ref = (ref.predicted_age - ref.actual_age).abs().values
        for m in ['amage'] + BASELINES + [x for x in ABLATION if x not in BASELINES] + ['cnn']:
            path = os.path.join(config.RESULTS_DIR, f'{m}_{cohort}', 'summary.json')
            if not os.path.exists(path):
                continue
            s = json.load(open(path))
            r = dict(cohort=cohort, model=m, **{k: v[0] for k, v in s.items()}, **{f'{k}_sd': v[1] for k, v in s.items()})
            if m != 'amage':
                x = preds(f'{m}_{cohort}'); assert (x.index == ref.index).all(), 'models must share the same test participants'
                r.update(paired((x.predicted_age - x.actual_age).abs().values - e_ref))
            rows.append(r)
    df = pd.DataFrame(rows)
    for cohort in df.cohort.unique():
        mk = (df.cohort == cohort) & df.model.isin(BASELINES)
        df.loc[mk, 'q'] = bh(df.loc[mk, 'p'])
    df.to_csv(os.path.join(config.ANALYSIS_DIR, 'comparison.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df[['cohort', 'model', 'mae', 'mae_sd', 'rmse', 'r2', 'pcc', 'dMAE', 'lo', 'hi', 'p', 'q']].round(3).to_string())


if __name__ == '__main__':
    main()
