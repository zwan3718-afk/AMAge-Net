"""Diagnostic: node representations of the GAT branch become identical across ROIs on the fully connected graph,
so the attention-pooling scores s_i (Eq. 6) are not informative about regions (see Methods).

Prints, for a batch of held-out participants, the across-node SD of the node embeddings relative to their mean
magnitude after each GAT layer, and the within-participant range of s_i.
Usage: python analysis/check_gat_oversmoothing.py --cohort camcan --fold 1
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from amagenet import data as D  # noqa: E402
from amagenet.models import AMAgeNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', default='camcan'); ap.add_argument('--fold', type=int, default=1)
    a = ap.parse_args()
    items, ages = D.subject_list(a.cohort); byid = {it[0]: it for it in items}
    sp = json.load(open(os.path.join(config.RESULTS_DIR, 'splits', a.cohort, f'fold{a.fold}.json')))
    m = AMAgeNet().cuda().eval()
    m.load_state_dict(torch.load(os.path.join(config.RESULTS_DIR, f'amage_{a.cohort}', f'fold{a.fold}_best.pt')))
    b = next(iter(DataLoader(D.BrainDataset(a.cohort, [byid[s] for s in sp['test'][:8]], {s: 0.0 for s in ages}), batch_size=8))).cuda()
    g = m.gcn
    with torch.no_grad():
        x0 = b.x; x1 = F.elu(g.gat1(x0, b.edge_index)); x2 = F.elu(g.gat2(x1, b.edge_index)); x3 = F.elu(g.gat_out(x2, b.edge_index))
        s = g.att_pool(x3).view(b.num_graphs, -1)
    for name, x in [('input', x0), ('GAT layer 1', x1), ('GAT layer 2', x2), ('GAT layer 3', x3)]:
        xs = x.view(b.num_graphs, 90, -1)
        print(f'{name:12s} across-node SD / mean|x| = {(xs.std(1).mean() / xs.abs().mean()).item():.2e}')
    print('edges per graph:', b.edge_index.shape[1] // b.num_graphs)
    print('within-participant range of s_i:', (s.max(1).values - s.min(1).values).cpu().numpy())


if __name__ == '__main__':
    main()
