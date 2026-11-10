"""Five-fold cross-validation with an internal validation set, identical for every method.

  outer folds : KFold(5, shuffle=True, random_state=42) over the sorted participant list (test fold = held out)
  inner split : 10% of each training portion, stratified by age quintile (random_state = 42 + fold)
  training    : max. 50 epochs, no early stopping; ReduceLROnPlateau (factor 0.5, patience 5, min 1e-6)
                monitors the internal-validation loss; the checkpoint with the lowest internal-validation MAE is kept
  testing     : the held-out fold is evaluated once with that checkpoint
  age scaling : min-max fitted on the training set of each fold

Usage:
  python train.py --model amage --cohort camcan          # see registry.py for all model names
  python train.py --model amage --cohort camcan --smoke  # tiny run to check the setup
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # MFFormer needs ~23.5 GB at batch size 8

import argparse
import json
import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import config
from amagenet import data as D
from registry import specs

SEED, EPOCHS, BATCH, LR, FOLDS, INNER_VAL = 42, 50, 8, 1e-4, 5, 0.10


def make_splits(cohort):
    """Create (or load) the shared splits for a cohort."""
    out = os.path.join(config.RESULTS_DIR, 'splits', cohort)
    if os.path.exists(os.path.join(out, f'fold{FOLDS}.json')):
        return [json.load(open(os.path.join(out, f'fold{k}.json'))) for k in range(1, FOLDS + 1)]
    os.makedirs(out, exist_ok=True)
    items, ages = D.subject_list(cohort)
    splits = []
    for k, (trv_idx, te_idx) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(items), start=1):
        trv = [items[i] for i in trv_idx]
        bins = pd.qcut(np.array([ages[s] for s, _, _ in trv]), q=5, labels=False, duplicates='drop')
        tr, va = train_test_split(trv, test_size=INNER_VAL, random_state=SEED + k, stratify=bins)
        sp = dict(train=[s for s, _, _ in tr], val=[s for s, _, _ in va], test=[items[i][0] for i in te_idx])
        json.dump(sp, open(os.path.join(out, f'fold{k}.json'), 'w'))
        splits.append(sp)
    return splits


class DSet(torch.utils.data.Dataset):
    def __init__(self, spec, cohort, sids, byid, ynorm):
        self.spec, self.cohort, self.sids, self.byid, self.ynorm = spec, cohort, sids, byid, ynorm

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        s = self.sids[i]
        d = self.spec.make(self.cohort, *self.byid[s])
        y = torch.tensor([self.ynorm[s]], dtype=torch.float32)
        if self.spec.kind == 'pyg':
            from torch_geometric.data import Data
            return Data(**d, y=y, subject_id=s)
        return d, y, s


def loader(spec, cohort, sids, byid, ynorm, shuffle):
    ds = DSet(spec, cohort, sids, byid, ynorm)
    if spec.kind == 'pyg':
        from torch_geometric.loader import DataLoader
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=4)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=4)


def unpack(spec, b, dev):
    if spec.kind == 'pyg':
        b = b.to(dev)
        return b, b.y.view(-1), list(b.subject_id)
    d, y, s = b
    return {k: v.to(dev) for k, v in d.items()}, y.view(-1).to(dev), list(s)


def run_epoch(spec, model, ld, dev, crit, opt=None, amp=None):
    train = opt is not None
    model.train(train)
    tot, P, Y, S = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for b in ld:
            b, y, s = unpack(spec, b, dev)
            with torch.autocast('cuda', enabled=spec.amp):
                p = spec.fwd(model, b).float()
                loss = crit(p, y)
            if train:
                opt.zero_grad()
                if amp:
                    amp.scale(loss).backward(); amp.step(opt); amp.update()
                else:
                    loss.backward(); opt.step()
            tot += loss.item() * len(y); P += p.detach().cpu().tolist(); Y += y.cpu().tolist(); S += s
    return tot / len(ld.dataset), np.array(P), np.array(Y), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(specs()))
    ap.add_argument('--cohort', required=True, choices=sorted(config.DATASETS))
    ap.add_argument('--smoke', action='store_true', help='1 epoch on a tiny subset, separate output folder')
    a = ap.parse_args()
    spec = specs()[a.model]
    items, ages = D.subject_list(a.cohort)
    byid = {s: (s, t1, fc) for s, t1, fc in items}
    splits = make_splits(a.cohort)
    out = os.path.join(config.RESULTS_DIR + ('_smoke' if a.smoke else ''), f'{a.model}_{a.cohort}')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda')
    for k in ([1] if a.smoke else range(1, FOLDS + 1)):
        if os.path.exists(os.path.join(out, f'fold{k}_test_predictions.csv')):
            continue
        tr, va, te = splits[k - 1]['train'], splits[k - 1]['val'], splits[k - 1]['test']
        if a.smoke:
            tr, va, te = tr[:40], va[:8], te[:8]
        lo, hi = min(ages[s] for s in tr), max(ages[s] for s in tr)
        yn = {s: (v - lo) / (hi - lo) for s, v in ages.items()}
        den = lambda v: np.asarray(v) * (hi - lo) + lo
        random.seed(SEED + k); np.random.seed(SEED + k); torch.manual_seed(SEED + k); torch.cuda.manual_seed_all(SEED + k)
        model = spec.build().to(dev)
        if spec.lazy_fix:  # build lazily created layers before the optimizer collects the parameters
            with torch.no_grad():
                spec.fwd(model, unpack(spec, next(iter(loader(spec, a.cohort, va[:2], byid, yn, False))), dev)[0])
        Opt = torch.optim.AdamW if spec.opt == 'adamw' else torch.optim.Adam
        opt = Opt(model.parameters(), lr=LR, weight_decay=spec.wd)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5, min_lr=1e-6)
        crit = torch.nn.L1Loss() if spec.loss == 'l1' else torch.nn.MSELoss()
        amp = torch.amp.GradScaler('cuda') if spec.amp else None
        Ltr, Lva, Lte = (loader(spec, a.cohort, x, byid, yn, sh) for x, sh in [(tr, True), (va, False), (te, False)])
        best, hist, t0 = dict(mae=np.inf), [], time.time()
        for ep in range(1, (1 if a.smoke else EPOCHS) + 1):
            trl, *_ = run_epoch(spec, model, Ltr, dev, crit, opt, amp)
            vl, vp, vy, _ = run_epoch(spec, model, Lva, dev, crit)
            vm = mean_absolute_error(den(vy), den(vp))
            sch.step(vl)                                   # internal validation only
            if vm < best['mae']:
                best = dict(mae=vm, epoch=ep, state=deepcopy(model.state_dict()))
            hist.append(dict(epoch=ep, train_loss=trl, val_loss=vl, val_mae=vm, lr=opt.param_groups[0]['lr']))
            print(f'{a.model} {a.cohort} fold {k} epoch {ep:02d}  train {trl:.4f}  val {vl:.4f}  val MAE {vm:.3f}', flush=True)
        pd.DataFrame(hist).to_csv(os.path.join(out, f'fold{k}_history.csv'), index=False)
        torch.save(best['state'], os.path.join(out, f'fold{k}_best.pt'))
        model.load_state_dict(best['state'])
        _, tp, ty, ts = run_epoch(spec, model, Lte, dev, crit)   # test fold: evaluated once
        tp, ty = den(tp), den(ty)
        pd.DataFrame(dict(subject_id=ts, fold=k, actual_age=ty, predicted_age=tp)).to_csv(
            os.path.join(out, f'fold{k}_test_predictions.csv'), index=False)
        m = dict(mae=float(mean_absolute_error(ty, tp)), rmse=float(np.sqrt(mean_squared_error(ty, tp))),
                 r2=float(r2_score(ty, tp)), pcc=float(pearsonr(ty, tp)[0]), best_epoch=best['epoch'],
                 minutes=(time.time() - t0) / 60)
        json.dump(m, open(os.path.join(out, f'fold{k}_metrics.json'), 'w'), indent=1)
        print(f'>>> {a.model} {a.cohort} fold {k}: test MAE {m["mae"]:.3f}, r {m["pcc"]:.3f} (epoch {best["epoch"]})', flush=True)
        del model, opt
        torch.cuda.empty_cache()
    if not a.smoke:
        per = [json.load(open(os.path.join(out, f'fold{k}_metrics.json'))) for k in range(1, FOLDS + 1)]
        summ = {k: (float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))) for k in ['mae', 'rmse', 'r2', 'pcc']}
        json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
        print('SUMMARY', a.model, a.cohort, summ, flush=True)


if __name__ == '__main__':
    main()
