"""Age bias and calibration of AMAge-Net (Table 2): slope/intercept of predicted on chronological age, correlation of
the brain age gap (BAG) with age, age-stratified errors, before and after a linear bias correction whose parameters
are estimated on the internal validation set of each fold.

Usage: python analysis/calibration.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from amagenet import data as D  # noqa: E402
from amagenet.models import AMAgeNet  # noqa: E402

STRATA = {'camcan': [18, 40, 60, 80, 100], 'oasis': [40, 60, 70, 80, 100]}


def val_predictions(cohort):
    items, ages = D.subject_list(cohort); byid = {it[0]: it for it in items}
    out = []
    for k in range(1, 6):
        sp = json.load(open(os.path.join(config.RESULTS_DIR, 'splits', cohort, f'fold{k}.json')))
        tr = np.array([ages[s] for s in sp['train']]); lo, hi = tr.min(), tr.max()
        m = AMAgeNet().cuda().eval()
        m.load_state_dict(torch.load(os.path.join(config.RESULTS_DIR, f'amage_{cohort}', f'fold{k}_best.pt')))
        ld = DataLoader(D.BrainDataset(cohort, [byid[s] for s in sp['val']], {s: 0.0 for s in ages}), batch_size=8)
        with torch.no_grad():
            for b in ld:
                p = m(b.cuda()).cpu().numpy() * (hi - lo) + lo
                out += [(k, s, ages[s], v) for s, v in zip(b.subject_id, p)]
    return pd.DataFrame(out, columns=['fold', 'subject_id', 'actual_age', 'predicted_age'])


def bias(age, pred, label):
    bag = pred - age
    sl, ic, *_ = stats.linregress(age, pred)
    r, p = stats.pearsonr(bag, age)
    return dict(predictions=label, slope=sl, intercept=ic, r_bag_age=r, p_bag_age=p, mae=np.abs(bag).mean(), mean_bag=bag.mean())


def main():
    rows, strata = [], []
    for cohort in config.DATASETS:
        te = pd.concat([pd.read_csv(os.path.join(config.RESULTS_DIR, f'amage_{cohort}', f'fold{k}_test_predictions.csv')) for k in range(1, 6)])
        va = val_predictions(cohort)
        te['corrected'] = np.nan
        for k in range(1, 6):
            v = va[va.fold == k]; a, b, *_ = stats.linregress(v.actual_age, v.predicted_age)
            te.loc[te.fold == k, 'corrected'] = (te.loc[te.fold == k, 'predicted_age'] - b) / a
        for lab, col in [('uncorrected', 'predicted_age'), ('corrected', 'corrected')]:
            rows.append(dict(cohort=cohort, **bias(te.actual_age.values, te[col].values, lab)))
        te['stratum'] = pd.cut(te.actual_age, STRATA[cohort], right=False)
        for st, g in te.groupby('stratum', observed=True):
            strata.append(dict(cohort=cohort, stratum=str(st), n=len(g),
                               mae_unc=np.abs(g.predicted_age - g.actual_age).mean(), bag_unc=(g.predicted_age - g.actual_age).mean(),
                               mae_cor=np.abs(g.corrected - g.actual_age).mean(), bag_cor=(g.corrected - g.actual_age).mean()))
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(config.ANALYSIS_DIR, 'calibration.csv'), index=False)
    pd.DataFrame(strata).to_csv(os.path.join(config.ANALYSIS_DIR, 'age_strata.csv'), index=False)
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(strata).round(2).to_string())


if __name__ == '__main__':
    main()
