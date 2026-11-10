# SET_GFRN.py

import torch
import torch.nn as nn

from .fMRI_Encoder import GraphFrequencyRecurrentNetwork
from .sMRI_Encoder import SETEncoder


class SETGFRN(nn.Module):

    def __init__(self):

        super().__init__()

        # sMRI Encoder
        self.smri_encoder = SETEncoder(
            in_channels=1,
            feature_dim=128
        )

        # fMRI Encoder
        self.fmri_encoder = GraphFrequencyRecurrentNetwork(
            num_rois=90,
            hidden_dim=128
        )

        # Fusion
        self.fusion = nn.Sequential(

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, smri_vol, bold_signal):

        smri_feat = self.smri_encoder(smri_vol)

        fmri_feat = self.fmri_encoder(bold_signal)

        fusion_feat = torch.cat(
            [smri_feat, fmri_feat],
            dim=1
        )

        age = self.fusion(fusion_feat)

        return age