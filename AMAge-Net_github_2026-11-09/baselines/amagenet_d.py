import torch
from torch import nn
from monai.networks.nets import DenseNet121


class SmriModel(nn.Module):
    def __init__(self):
        super(SmriModel, self).__init__()

        # 保留 DenseNet121 的 features 部分，作为结构特征提取器
        self.backbone = DenseNet121(spatial_dims=3, in_channels=1, out_channels=64)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # 去掉 classifier

        # 投影为和融合模型一致的维度
        self.patch_proj = nn.Linear(1024, 64)

        # 最终回归层
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img):
        if img.dim() == 4:
            img = img.unsqueeze(1)  # [B, 1, D, H, W]
        B = img.size(0)

        # CNN 提取特征图
        feat_map = self.feature_extractor(img)               # [B, 1024, d, h, w]
        feat_tokens = feat_map.flatten(2).transpose(1, 2)    # [B, N, 1024]
        feat_tokens = self.patch_proj(feat_tokens)           # [B, N, 64]


        # 平均池化作为全局结构特征
        global_feat = torch.mean(feat_tokens, dim=1)         # [B, 64]

        out = self.fc(global_feat).squeeze(1)                # [B]
        return out

