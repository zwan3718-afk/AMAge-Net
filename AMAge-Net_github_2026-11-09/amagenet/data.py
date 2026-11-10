"""Subject lists, T1w preprocessing (cached) and the multimodal graph dataset."""
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import config

TARGET_SHAPE = (128, 128, 128)


def subject_list(cohort):
    """Participants with a T1w volume, an FC matrix, and an age. Returns ([(sid, t1, fc)], {sid: age})."""
    c = config.DATASETS[cohort]
    df = pd.read_csv(c['ages'], encoding='gbk')
    df['Subject'] = df['Subject'].astype(str).str.strip()
    ages = dict(zip(df['Subject'], df['Age'].astype(float)))
    items = []
    for t1 in sorted(glob.glob(c['t1'].format(sid='*'))):
        sid = os.path.basename(os.path.dirname(t1)).strip()
        fc = c['fc'].format(sid=sid)
        if sid in ages and os.path.exists(fc):
            items.append((sid, t1, fc))
    return items, ages


def sex_table(cohort):
    path = config.DATASETS[cohort].get('sex')
    if not path:
        return {}
    df = pd.read_csv(path, encoding='gbk')
    return dict(zip(df['Subject'].astype(str).str.strip(), df['Sex'].str.upper()))


def t1_volume(cohort, sid, path, shape=TARGET_SHAPE):
    """Linear resampling to `shape` and min-max scaling to [0, 1]; cached as .npy."""
    tag = cohort if shape == TARGET_SHAPE else f'{cohort}_{shape[0]}'
    cpath = os.path.join(config.CACHE_DIR, tag, f'{sid}.npy')
    if os.path.exists(cpath):
        return np.load(cpath)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    img = nib.load(path).get_fdata()
    if shape != TARGET_SHAPE:
        img = img.astype(np.float32)
    img = zoom(img, [t / o for t, o in zip(shape, img.shape)], order=1)
    img = ((img - img.min()) / (img.max() - img.min() + (1e-8 if shape != TARGET_SHAPE else 0))).astype(np.float32)
    tmp = cpath + f'.tmp{os.getpid()}.npy'
    np.save(tmp, img)
    os.replace(tmp, cpath)
    return img


def fc_matrix(path):
    return pd.read_csv(path, index_col=0).values.astype(np.float32)


def graph_fields(fc, braingnn=False):
    """Node features = FC rows; edges from |FC| != 0 (fully connected); |FC| is kept as edge_attr only for BrainGNN."""
    conn = torch.tensor(fc)
    edge_index, edge_attr = dense_to_sparse(conn.abs())
    d = dict(x=conn.clone(), edge_index=edge_index, edge_attr=edge_attr)
    if braingnn:
        d.update(edge_attr=edge_attr.view(-1, 1), pos=torch.zeros(conn.size(0), 3))
    return d


class BrainDataset(Dataset):
    """PyG Data with x (90x90 FC rows), edge_index, img (1x128^3) and y (normalized age)."""
    def __init__(self, cohort, items, age_norm):
        self.cohort, self.items, self.age_norm = cohort, items, age_norm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, t1, fc = self.items[idx]
        img = torch.from_numpy(t1_volume(self.cohort, sid, t1)).unsqueeze(0)
        return Data(**graph_fields(fc_matrix(fc)), img=img,
                    y=torch.tensor([self.age_norm[sid]], dtype=torch.float32), subject_id=sid)
