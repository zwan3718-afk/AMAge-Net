"""Model registry: architecture, input construction, loss and optimizer for every method in the paper.

Each competing method keeps its original architecture, input preprocessing, loss and optimizer;
all methods share the same splits and model-selection procedure (see train.py).
"""
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from amagenet import data as D
from amagenet.models import AMAgeNet, AMAgeNetConcat, AMAgeNetCross, AMAgeNetGate


class Spec:
    """kind: 'pyg' -> torch_geometric Data batches; 'tensor' -> dict of tensors.
    make(cohort, sid, t1, fc) -> dict of inputs; build() -> nn.Module; fwd(model, batch) -> predictions [B]."""
    def __init__(self, build, make, fwd, kind='tensor', loss='mse', opt='adam', wd=0.0, amp=False, lazy_fix=True):
        self.build, self.make, self.fwd, self.kind = build, make, fwd, kind
        self.loss, self.opt, self.wd, self.amp, self.lazy_fix = loss, opt, wd, amp, lazy_fix


def _flat(o):
    o = o[0] if isinstance(o, (tuple, list)) else o
    return o.reshape(-1)


def _graph(c, s, t1, fc, img=True):
    d = D.graph_fields(D.fc_matrix(fc))
    if img:
        d['img'] = torch.from_numpy(D.t1_volume(c, s, t1)).unsqueeze(0)
    return d


def _t1(c, s, t1, fc):
    return dict(img=torch.from_numpy(D.t1_volume(c, s, t1)).unsqueeze(0))


def _ctransformer(c, s, t1, fc):
    img = D.t1_volume(c, s, t1)                       # z-score is invariant to the min-max scaling
    img = (img - img.mean()) / (img.std() + 1e-6)
    f = np.arctanh(np.clip(D.fc_matrix(fc), -0.999, 0.999)).astype(np.float32)
    return dict(smri=torch.from_numpy(img.astype(np.float32)).unsqueeze(0), fc=torch.from_numpy(f))


def _setgfrn(c, s, t1, fc):
    x = D.fc_matrix(fc)                               # as in the original SET-GFRN pipeline (FC matrix input)
    nyq = (1 / 0.72) / 2
    b, a = butter(4, [0.01 / nyq, 0.1 / nyq], btype='band')
    x = filtfilt(b, a, x, axis=0).copy().astype(np.float32)
    return dict(smri=torch.from_numpy(D.t1_volume(c, s, t1, shape=(96, 96, 96))).unsqueeze(0), bold=torch.from_numpy(x))


def _mfformer(c, s, t1, fc):
    f = np.nan_to_num(D.fc_matrix(fc), nan=0.0, posinf=1.0, neginf=-1.0)
    return dict(t1w=torch.from_numpy(D.t1_volume(c, s, t1)).unsqueeze(0), fc=torch.from_numpy(f).unsqueeze(0))


def _cnn_wrapper():
    from baselines.cnn3d import CNN

    class CNNWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = CNN()

        def forward(self, data):
            img = data.img
            return self.net(img.unsqueeze(1) if img.dim() == 4 else img).squeeze(1)
    return CNNWrapper()


def specs():
    from baselines import braingnn, gcn, bc_gcn, amagenet_g, amagenet_d, resnet3d, efficientnet3d, mfformer, ctransformer
    from baselines.set_gfrn.model import SETGFRN
    pyg = dict(kind='pyg')
    return {
        # proposed model and fusion ablations (Adam, weight decay 1e-4, MSE)
        'amage': Spec(AMAgeNet, _graph, lambda m, b: _flat(m(b)), wd=1e-4, **pyg),
        'concat': Spec(AMAgeNetConcat, _graph, lambda m, b: _flat(m(b)), wd=1e-4, **pyg),
        'cross': Spec(AMAgeNetCross, _graph, lambda m, b: _flat(m(b)), wd=1e-4, **pyg),
        'gate': Spec(AMAgeNetGate, _graph, lambda m, b: _flat(m(b)), wd=1e-4, **pyg),
        # single-modality ablations
        'g': Spec(amagenet_g.FMRIModel, lambda c, s, t1, fc: _graph(c, s, t1, fc, img=False), lambda m, b: _flat(m(b)), **pyg),
        'd': Spec(amagenet_d.SmriModel, _t1, lambda m, b: _flat(m(b['img']))),
        # fMRI baselines
        'braingnn': Spec(lambda: braingnn.Network(indim=90, ratio=0.8, gnn_hid=128, nclass=1),
                         lambda c, s, t1, fc: D.graph_fields(D.fc_matrix(fc), braingnn=True),
                         lambda m, b: _flat(m(b.x, b.edge_index, b.batch, b.edge_attr, b.pos)), **pyg),
        'gcn': Spec(lambda: gcn.GraphCNN(input_dim=90, hidden_dim=64, output_dim=1, num_gc_layers=2, mlp_layers=2),
                    lambda c, s, t1, fc: D.graph_fields(D.fc_matrix(fc)), lambda m, b: _flat(m(b.x, b.edge_index, b.batch)), **pyg),
        'bc_gcn': Spec(lambda: bc_gcn.BC_GCN_SE(16, 16, 16, 64, 256),
                       lambda c, s, t1, fc: dict(x=torch.from_numpy(D.fc_matrix(fc)).unsqueeze(0)), lambda m, b: _flat(m(b['x']))),
        # sMRI baselines
        'resnet': Spec(resnet3d.SmriModel, _t1, lambda m, b: _flat(m(b['img']))),
        'efficientnet': Spec(efficientnet3d.SmriModel, _t1, lambda m, b: _flat(m(b['img']))),
        'cnn_orig': Spec(_cnn_wrapper, _graph, lambda m, b: _flat(m(b)), lazy_fix=False, **pyg),   # original 3D-CNN settings
        'cnn': Spec(_cnn_wrapper, _graph, lambda m, b: _flat(m(b)), wd=1e-4, **pyg),                # unified settings (Table S14)
        # multimodal baselines (AdamW, weight decay 1e-4, L1 loss)
        'mfformer': Spec(mfformer.MFFormerAge, _mfformer, lambda m, b: _flat(m(b['fc'], b['t1w'])), loss='l1', opt='adamw', wd=1e-4),
        'ctransformer': Spec(ctransformer.MultimodalBrainAgeModel, _ctransformer, lambda m, b: _flat(m(b['smri'], b['fc'])), loss='l1', opt='adamw', wd=1e-4),
        'setgfrn': Spec(SETGFRN, _setgfrn, lambda m, b: _flat(m(b['smri'], b['bold'])), loss='l1', opt='adamw', wd=1e-4, amp=True),
    }
