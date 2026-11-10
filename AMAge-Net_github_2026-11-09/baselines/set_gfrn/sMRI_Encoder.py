import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# SE Attention Block
# ======================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Attention
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),

            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _, _ = x.size()

        # Global pooling
        y = self.avg_pool(x).view(b, c)

        # Channel attention
        y = self.fc(y).view(b, c, 1, 1, 1)

        return x * y


# ======================================================
# Conv Block
# ======================================================
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ======================================================
# SET Encoder
# ======================================================
class SETEncoder(nn.Module):
    """
    Structure Encoder Transformer

    Input:
        [B, 1, D, H, W]

    Output:
        [B, feature_dim]
    """

    def __init__(
            self,
            in_channels=1,
            feature_dim=256,
            transformer_dim=256,
            num_heads=4,
            num_layers=2,
            dropout=0.1
    ):

        super(SETEncoder, self).__init__()

        # --------------------------------------------------
        # CNN Backbone
        # --------------------------------------------------
        self.conv1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = ConvBlock(32, 64)
        self.se1 = SEBlock(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = ConvBlock(64, 128)
        self.se2 = SEBlock(128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = ConvBlock(128, transformer_dim)

        # --------------------------------------------------
        # Transformer Encoder
        # --------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # --------------------------------------------------
        # Final Projection
        # --------------------------------------------------
        self.fc = nn.Sequential(

            nn.Linear(transformer_dim, feature_dim),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout)
        )

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x):

        # ---------------------------------------------
        # CNN Feature Extraction
        # ---------------------------------------------
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.se1(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.se2(x)
        x = self.pool3(x)

        x = self.conv4(x)

        # x shape:
        # [B, C, D, H, W]

        B, C, D, H, W = x.shape

        # ---------------------------------------------
        # Flatten into Tokens
        # ---------------------------------------------
        x = x.view(B, C, -1)

        # [B, N, C]
        x = x.permute(0, 2, 1)

        # ---------------------------------------------
        # Transformer
        # ---------------------------------------------
        x = self.transformer(x)

        # ---------------------------------------------
        # Global Average Pooling
        # ---------------------------------------------
        x = x.mean(dim=1)

        # ---------------------------------------------
        # Final Embedding
        # ---------------------------------------------
        x = self.fc(x)

        return x


