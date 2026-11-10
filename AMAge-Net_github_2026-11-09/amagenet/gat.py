"""Graph attention branch of AMAge-Net (fMRI).

Node features are the rows of the (signed) FC matrix; the graph topology is defined by |FC| != 0,
which yields a fully connected graph. |FC| is NOT passed as an edge attribute.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.1):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate, concat=True)
        # last layer: single head, no concatenation
        self.gat_out = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        # attention pooling (node score s_i in Eq. 6)
        self.att_pool = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = F.elu(self.gat_out(x, edge_index))  # [num_nodes, hidden_dim]
        attn_score = self.att_pool(x)           # [num_nodes, 1]
        x = global_mean_pool(x * attn_score, batch)  # [batch_size, hidden_dim]
        return self.regressor(x)                # [batch_size, output_dim]
