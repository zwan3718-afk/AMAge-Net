import torch
from torch import nn
from amagenet.gat import GATModel


class FMRIModel(nn.Module):
    def __init__(self, gcn_input_dim=90, gcn_hidden=64, gcn_out=64, num_heads=4, dropout_rate=0):
        super(FMRIModel, self).__init__()

        self.gcn = GATModel(
            input_dim=gcn_input_dim,
            hidden_dim=gcn_hidden,
            output_dim=gcn_out,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        self.fc = nn.Sequential(
            nn.Linear(gcn_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):

        gcn_feat = self.gcn(data.x, data.edge_index, data.batch)  # [batch_size, gcn_out]
        out = self.fc(gcn_feat).squeeze(1)  # [batch_size]
        return out

