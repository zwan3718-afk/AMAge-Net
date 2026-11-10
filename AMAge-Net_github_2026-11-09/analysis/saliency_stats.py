"""Regional saliency statistics and rendering data (Figures 6-9, Tables S4-S7, S9, S10, S12, S13).

  ranking      : group mean (fMRI: per-participant relative contribution; sMRI: raw |gradient|), min-max over 90 regions
  display rule : top quartile (23 of 90 regions) for both modalities
  stability    : 1,000 bootstrap resamples of participants (rank intervals, top-k frequency)
  replication  : Spearman correlation and top-k overlap between Cam-CAN and OASIS-3
  sex          : common model; region-wise one-way ANOVA (ranks / normalized saliency), global max-F permutation
                 test (5,000 permutations of the sex labels), BH-FDR over the 90 regions
  rendering    : per figure, a CSV (all regions) and a NIfTI map (AAL regions filled with their score)

Usage: python analysis/saliency_stats.py      (after analysis/saliency.py)
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from amagenet import data as D  # noqa: E402

K = 23  # top quartile of the 90 AAL regions
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = config.ANALYSIS_DIR
PKG = os.path.join(OUT, 'rendering')
rng = np.random.default_rng(0)
mm = lambda v: (v - v.min()) / (v.max() - v.min())


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for r, i in list(enumerate(o))[::-1]:
        prev = min(prev, p[i] * n / (r + 1)); q[i] = prev
    return q


def load(modality, cohort, code2name):
    d = pd.read_csv(os.path.join(OUT, f'saliency_{modality}_{cohort}.csv'))
    if modality == 'fmri':
        regs = [code2name[int(float(c))] for c in d.columns[1:]]
        V = d.drop(columns='subject_id').values
        return d.subject_id.astype(str).values, regs, V / V.sum(1, keepdims=True)
    regs = list(d.columns[1:])
    return d.subject_id.astype(str).values, regs, d[regs].values


def main():
    os.makedirs(PKG, exist_ok=True)
    lab = pd.read_csv(config.AAL_LABELS, sep=r'\s+', header=None, names=['Abbr', 'Name', 'Index'])[:90]
    code2name = dict(zip(lab.Index, lab.Name)); name2idx = dict(zip(lab.Name, lab.Index))
    abbr = dict(pd.read_csv(os.path.join(os.path.dirname(HERE), 'data', 'aal90_abbreviations.csv')).values)
    aal_img = nib.load(config.AAL_NII); aal = aal_img.get_fdata().astype(int)
    sex = D.sex_table('camcan')
    summary = {}
    for mod in ['fmri', 'smri']:
        sid, regs, V = load(mod, 'camcan', code2name)
        score = pd.Series(mm(V.mean(0)), index=regs)
        top = list(score.sort_values(ascending=False).index[:K])
        means, ranks = [], []
        for _ in range(1000):
            m = V[rng.integers(0, len(V), len(V))].mean(0); means.append(mm(m))
            r = np.empty(len(regs)); r[np.argsort(m)[::-1]] = np.arange(1, len(regs) + 1); ranks.append(r)
        means, ranks = np.array(means), np.array(ranks)
        pd.DataFrame(dict(region=regs, score=score.values, rank=score.rank(ascending=False, method='first').astype(int).values,
                          boot_mean=means.mean(0), boot_sd=means.std(0), rank_mean=ranks.mean(0), rank_sd=ranks.std(0),
                          rank_lo=np.percentile(ranks, 2.5, 0), rank_hi=np.percentile(ranks, 97.5, 0),
                          top10=(ranks <= 10).mean(0), top1=(ranks == 1).mean(0), topK=(ranks <= K).mean(0))
                     ).sort_values('rank').to_csv(os.path.join(OUT, f'saliency_{mod}_table.csv'), index=False)
        res = dict(top=top)
        if os.path.exists(os.path.join(OUT, f'saliency_{mod}_oasis.csv')):
            _, regs_o, Vo = load(mod, 'oasis', code2name)
            so = pd.Series(mm(Vo.mean(0)), index=regs_o).loc[regs]
            res['cross_cohort_rho'] = float(stats.spearmanr(score, so)[0])
            res['sensitivity'] = []
            for k in sorted({10, 15, 20, K, 30, 40}):
                sa, sb = set(score.nlargest(k).index), set(so.nlargest(k).index)
                res['sensitivity'].append(dict(k=k, overlap_with_reported=len(sa & set(top)) / len(sa | set(top)),
                                               cross_cohort_jaccard=len(sa & sb) / len(sa | sb)))
        # sex differences with the common model
        g = np.array([sex.get(s) for s in sid]); ok = np.isin(g, ['MALE', 'FEMALE'])
        X, male = V[ok], g[ok] == 'MALE'
        Xn = X / X.sum(1, keepdims=True); Xr = np.apply_along_axis(lambda v: stats.rankdata(-v), 1, X)
        res['sex'] = {}
        for meas, M in [('rank', Xr), ('normalized', Xn)]:
            F = stats.f_oneway(M[male], M[~male]).statistic
            null = np.empty(5000); pr = np.zeros(len(regs))
            for i in range(5000):
                pm = rng.permutation(male); Fp = stats.f_oneway(M[pm], M[~pm]).statistic
                null[i] = Fp.max(); pr += Fp >= F
            q = bh((pr + 1) / 5001)
            pd.DataFrame(dict(region=regs, F=F, p=(pr + 1) / 5001, q=q, male_minus_female=Xn[male].mean(0) - Xn[~male].mean(0))
                         ).sort_values('p').to_csv(os.path.join(OUT, f'saliency_{mod}_sex_{meas}.csv'), index=False)
            res['sex'][meas] = dict(maxF=float(F.max()), region=regs[int(F.argmax())], p_global=float((np.sum(null >= F.max()) + 1) / 5001),
                                    n_regions_q05=int((q < 0.05).sum()), rho=float(stats.spearmanr(M[male].mean(0), M[~male].mean(0))[0]))
        gm, gf = pd.Series(mm(X[male].mean(0)), index=regs), pd.Series(mm(X[~male].mean(0)), index=regs)
        res['sex']['topK_shared'] = len(set(gm.nlargest(K).index) & set(gf.nlargest(K).index))
        summary[mod] = res
        # rendering data
        fig = {'fmri': ('Fig6_fMRI', 'Fig8A_fMRI_female', 'Fig8B_fMRI_male'), 'smri': ('Fig7_sMRI', 'Fig9A_sMRI_female', 'Fig9B_sMRI_male')}[mod]
        for tag, sc in zip(fig, [score, gf, gm]):
            sel = set(sc.nlargest(K).index)
            pd.DataFrame(dict(AAL_index=[name2idx[r] for r in regs], AAL_name=regs,
                              abbreviation=[abbr[r.rsplit('_', 1)[0]] + '.' + r.rsplit('_', 1)[1] for r in regs],
                              score=sc.values.round(4), rank=sc.rank(ascending=False, method='first').astype(int).values,
                              displayed=[r in sel for r in regs])).sort_values('rank').to_csv(os.path.join(PKG, f'{tag}.csv'), index=False)
            vol = np.zeros(aal.shape, np.float32)
            for r in sel:
                vol[aal == name2idx[r]] = sc[r]
            nib.save(nib.Nifti1Image(vol, aal_img.affine), os.path.join(PKG, f'{tag}.nii.gz'))
    summary['cross_modal_shared'] = sorted(set(summary['fmri']['top']) & set(summary['smri']['top']))
    json.dump(summary, open(os.path.join(OUT, 'saliency_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == '__main__':
    main()
