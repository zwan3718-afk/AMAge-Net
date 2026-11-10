"""AMAge-Net and its fusion variants (ablation).

All variants share the sMRI branch (3D DenseNet121, MONAI) and the fMRI branch (GAT).
Attribute names are kept identical to the training code so that released checkpoints load directly.

Tensor shapes per participant (128^3 input):
  structural tokens F_s: 64 x 64  (4x4x4 feature map, 1024 -> 64 projection)
  functional vector F_f: 1 x 64   (graph-level GAT output, 32 -> 64 projection)
  cross-attention: query = F_f (length 1), keys/values = F_s, 4 heads -> attention 1 x 64 per head
  gated fusion: scalar gate g in (0, 1) per participant
"""
import torch
from torch import nn
from monai.networks.nets import DenseNet121

from .gat import GATModel


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        x = self.norm1(query + self.dropout1(attn_output))
        return self.norm2(x + self.dropout2(self.ffn(x)))  # [B, 1, 64]


class GatedFusion(nn.Module):
    """Scalar gate computed by a two-layer perceptron: g = sigmoid(w2^T ReLU(W1 [a; b] + b1) + b2)."""
    def __init__(self, embed_dim=64):
        super(GatedFusion, self).__init__()
        self.gate_layer = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, feat1, feat2):
        gate = self.gate_layer(torch.cat([feat1, feat2], dim=-1))  # [B, 1, 1]
        return gate * feat1 + (1 - gate) * feat2


class _Branches(nn.Module):
    """sMRI tokens + fMRI vector, shared by all fusion variants."""
    def _init_branches(self, gcn_input_dim, gcn_hidden, gcn_out, num_heads, dropout_rate):
        self.backbone = DenseNet121(spatial_dims=3, in_channels=1, out_channels=64)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # drop classification head
        self.patch_proj = nn.Linear(1024, 64)
        self.gcn = GATModel(input_dim=gcn_input_dim, hidden_dim=gcn_hidden, output_dim=gcn_out,
                            num_heads=num_heads, dropout_rate=dropout_rate)
        self.gcn_proj = nn.Linear(gcn_out, 64)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def tokens(self, data):
        img = data.img
        if img.dim() == 4:
            img = img.unsqueeze(1)                                          # [B, 1, D, H, W]
        t = self.feature_extractor(img).flatten(2).transpose(1, 2)          # [B, N, 1024]
        t = self.patch_proj(t)                                              # [B, N, 64]
        f = self.gcn_proj(self.gcn(data.x, data.edge_index, data.batch)).unsqueeze(1)  # [B, 1, 64]
        return t, f


class AMAgeNet(_Branches):
    """Full model: Multi-Head Cross-Attention followed by Gated Fusion."""
    def __init__(self, gcn_input_dim=90, gcn_hidden=64, gcn_out=32, num_heads=4, dropout_rate=0):
        super().__init__()
        self._init_branches(gcn_input_dim, gcn_hidden, gcn_out, num_heads, dropout_rate)
        self.cross_attn_block = CrossAttentionBlock(embed_dim=64, num_heads=num_heads, dropout=dropout_rate)
        self.gated_fusion = GatedFusion(embed_dim=64)

    def forward(self, data):
        t, f = self.tokens(data)
        z = self.cross_attn_block(query=f, key=t, value=t)                 # [B, 1, 64]
        fused = self.gated_fusion(z, torch.mean(t, dim=1, keepdim=True))   # [B, 1, 64]
        return self.fc(fused.squeeze(1)).squeeze(1)                        # [B]


class AMAgeNetCross(_Branches):
    """Ablation: Cross-Attention only (no Gated Fusion)."""
    def __init__(self, gcn_input_dim=90, gcn_hidden=64, gcn_out=32, num_heads=4, dropout_rate=0):
        super().__init__()
        self._init_branches(gcn_input_dim, gcn_hidden, gcn_out, num_heads, dropout_rate)
        self.cross_attn_block = CrossAttentionBlock(embed_dim=64, num_heads=num_heads, dropout=dropout_rate)

    def forward(self, data):
        t, f = self.tokens(data)
        return self.fc(self.cross_attn_block(query=f, key=t, value=t).squeeze(1)).squeeze(1)


class AMAgeNetGate(_Branches):
    """Ablation: Gated Fusion only (no Cross-Attention); gates the fMRI vector against the mean sMRI token."""
    def __init__(self, gcn_input_dim=90, gcn_hidden=64, gcn_out=32, num_heads=4, dropout_rate=0):
        super().__init__()
        self._init_branches(gcn_input_dim, gcn_hidden, gcn_out, num_heads, dropout_rate)
        self.gated_fusion = GatedFusion(embed_dim=64)

    def forward(self, data):
        t, f = self.tokens(data)
        return self.fc(self.gated_fusion(f, torch.mean(t, dim=1, keepdim=True)).squeeze(1)).squeeze(1)


class AMAgeNetConcat(nn.Module):
    """Baseline fusion: concatenation of the global sMRI (DenseNet121) and fMRI (GAT) representations."""
    def __init__(self, gcn_input_dim=90, gcn_hidden=64, gcn_out=32, dropout_rate=0):
        super().__init__()
        self.cnn = DenseNet121(spatial_dims=3, in_channels=1, out_channels=64)
        self.gcn = GATModel(input_dim=gcn_input_dim, hidden_dim=gcn_hidden, output_dim=gcn_out, num_heads=4, dropout_rate=dropout_rate)
        self.gcn_proj = nn.Linear(gcn_out, 64)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, data):
        img = data.img
        if img.dim() == 4:
            img = img.unsqueeze(1)
        g = self.gcn_proj(self.gcn(data.x, data.edge_index, data.batch))
        return self.fc(torch.cat([self.cnn(img), g], dim=1)).squeeze(1)


MODELS = {'amage': AMAgeNet, 'cross': AMAgeNetCross, 'gate': AMAgeNetGate, 'concat': AMAgeNetConcat}
