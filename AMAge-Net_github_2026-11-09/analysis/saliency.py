"""Per-participant regional saliency from the final AMAge-Net models, computed on each participant's held-out test fold.

  fMRI: gradient x input of the predicted age w.r.t. the FC node features, summed over each node's features
  sMRI: |d MSE / d voxel| of the input T1w volume, resampled to AAL space and averaged within each of the 90 regions

Writes <ANALYSIS_DIR>/saliency_fmri_<cohort>.csv and saliency_smri_<cohort>.csv (participants x 90 AAL regions).
Usage: python analysis/saliency.py
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from amagenet import data as D  # noqa: E402
from amagenet.models import AMAgeNet  # noqa: E402


def aal90():
    lab = pd.read_csv(config.AAL_LABELS, sep=r'\s+', header=None, names=['Abbr', 'Name', 'Index'])[:90]
    return lab, nib.load(config.AAL_NII).get_fdata().astype(int)


def main():
    lab, aal = aal90()
    masks = [aal == i for i in lab.Index]
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    for cohort in config.DATASETS:
        items, ages = D.subject_list(cohort); byid = {it[0]: it for it in items}
        fc_cols = pd.read_csv(items[0][2], index_col=0).columns.tolist()  # AAL codes in the FC files
        f_rows, s_rows = [], []
        for k in range(1, 6):
            sp = json.load(open(os.path.join(config.RESULTS_DIR, 'splits', cohort, f'fold{k}.json')))
            tr = np.array([ages[s] for s in sp['train']]); lo, hi = tr.min(), tr.max()
            yn = {s: (a - lo) / (hi - lo) for s, a in ages.items()}
            m = AMAgeNet().cuda().eval()
            m.load_state_dict(torch.load(os.path.join(config.RESULTS_DIR, f'amage_{cohort}', f'fold{k}_best.pt')))
            for b in DataLoader(D.BrainDataset(cohort, [byid[s] for s in sp['test']], yn), batch_size=1):
                b = b.cuda()
                b.x.requires_grad_(True); b.img.requires_grad_(True)
                pred = m(b)
                gx, = torch.autograd.grad(pred.sum(), b.x, retain_graph=True)
                f_rows.append([b.subject_id[0]] + (gx * b.x).abs().sum(1).detach().cpu().tolist())
                gi, = torch.autograd.grad(torch.nn.functional.mse_loss(pred, b.y.view(-1)), b.img)
                sal = gi.abs().squeeze().cpu().numpy()
                sal = zoom(sal, [a / s for a, s in zip(aal.shape, sal.shape)], order=1)
                s_rows.append([b.subject_id[0]] + [float(sal[mk].mean()) for mk in masks])
            print(cohort, 'fold', k, 'done', flush=True)
        pd.DataFrame(f_rows, columns=['subject_id'] + fc_cols).to_csv(os.path.join(config.ANALYSIS_DIR, f'saliency_fmri_{cohort}.csv'), index=False)
        pd.DataFrame(s_rows, columns=['subject_id'] + list(lab.Name)).to_csv(os.path.join(config.ANALYSIS_DIR, f'saliency_smri_{cohort}.csv'), index=False)


if __name__ == '__main__':
    main()
