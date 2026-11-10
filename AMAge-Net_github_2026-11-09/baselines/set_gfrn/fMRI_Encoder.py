# fMRI_Encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data

class GraphFrequencyRecurrentNetwork(nn.Module):

    def __init__(
            self,
            num_rois=90,
            hidden_dim=128,
            num_freq_bands=3,
            window_size=30,
            stride=5
    ):

        super().__init__()

        self.num_rois = num_rois
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.stride = stride

        # -----------------------------------------
        # GIN
        # -----------------------------------------
        self.gin_layers = nn.ModuleList()

        for _ in range(num_freq_bands):

            mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_dim)
            )

            self.gin_layers.append(
                GINConv(mlp, train_eps=True)
            )

        # -----------------------------------------
        # GRU
        # -----------------------------------------
        self.gru = nn.GRU(
            input_size=hidden_dim * num_freq_bands,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # -----------------------------------------
        # Temporal Attention
        # -----------------------------------------
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )

        # -----------------------------------------
        # Final Projection
        # -----------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    # ======================================================
    # Frequency decomposition
    # ======================================================
    def frequency_decomposition(self, signal):

        signal = signal.cpu().numpy()

        fs = 1 / 0.72
        T = signal.shape[-1]

        freq = np.fft.fftfreq(T, d=1/fs)
        fft_signal = np.fft.fft(signal, axis=-1)

        bands = []

        freq_ranges = [
            (0.01, 0.027),
            (0.027, 0.073),
            (0.073, 0.1)
        ]

        for low, high in freq_ranges:

            mask = (freq >= low) & (freq <= high)

            fft_band = fft_signal.copy()
            fft_band[..., ~mask] = 0

            band_signal = np.fft.ifft(
                fft_band,
                axis=-1
            ).real

            bands.append(
                torch.tensor(
                    band_signal,
                    dtype=torch.float32,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            )

        return bands

    # ======================================================
    # Build dynamic graph
    # ======================================================
    def build_graph(self, x):

        corr = torch.corrcoef(x)

        edge_index = torch.nonzero(
            corr > 0.5,
            as_tuple=False
        ).t()

        if edge_index.shape[1] == 0:
            edge_index = torch.arange(
                x.shape[0],
                device=x.device
            ).repeat(2,1)

        return edge_index

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, bold_signal):

        B, R, T = bold_signal.shape

        bands = self.frequency_decomposition(bold_signal)

        all_subject_features = []

        # ==================================================
        # For each subject
        # ==================================================
        for b in range(B):

            temporal_features = []

            # ----------------------------------------------
            # Sliding windows
            # ----------------------------------------------
            for t in range(
                    0,
                    T - self.window_size + 1,
                    self.stride
            ):

                band_features = []

                for band_idx, band_signal in enumerate(bands):

                    window_data = band_signal[
                        b,
                        :,
                        t:t+self.window_size
                    ]

                    edge_index = self.build_graph(window_data)

                    node_feat = window_data.mean(
                        dim=-1,
                        keepdim=True
                    )

                    x = self.gin_layers[band_idx](
                        node_feat,
                        edge_index
                    )

                    graph_feat = x.mean(dim=0)

                    band_features.append(graph_feat)

                band_features = torch.cat(
                    band_features,
                    dim=-1
                )

                temporal_features.append(
                    band_features
                )

            temporal_features = torch.stack(
                temporal_features,
                dim=0
            )

            all_subject_features.append(
                temporal_features
            )

        # [B, num_windows, feature]
        temporal_features = torch.stack(
            all_subject_features,
            dim=0
        )

        # ==================================================
        # GRU
        # ==================================================
        gru_out, _ = self.gru(temporal_features)

        # ==================================================
        # Attention
        # ==================================================
        attn_out, _ = self.attention(
            gru_out,
            gru_out,
            gru_out
        )

        # ==================================================
        # Global pooling
        # ==================================================
        features = attn_out.mean(dim=1)

        # ==================================================
        # Projection
        # ==================================================
        features = self.fc(features)

        return features