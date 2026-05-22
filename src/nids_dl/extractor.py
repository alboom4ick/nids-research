"""DL feature extractor: CNN-MCL -> BI-LSTM -> FWA (BI-ATT).

Produces Fe = concat(FBL, FAT) per Algorithm 3 (paper line 482). The forward
pass is `Fe, aux = extractor(x)`; during phase-1 training a temporary softmax
head consumes Fe to provide a differentiable loss for Adam (see train.py).
"""
from __future__ import annotations

import torch
from torch import nn

from .layers import BiLSTM, FWA, MCL


class Extractor(nn.Module):
    def __init__(
        self,
        in_features: int = 120,
        mcl_filters: int = 10,
        wc_dim: int = 56,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        attn_dim: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.mcl = MCL(
            in_features=in_features,
            n_filters=mcl_filters,
            out_features=wc_dim,
        )
        self.bilstm = BiLSTM(
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.fwa = FWA(
            hidden_dim=2 * lstm_hidden,
            wc_dim=wc_dim,
            attn_dim=attn_dim,
        )
        self.feature_dim = 4 * lstm_hidden

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        wc = self.mcl(x)                                       # (N, wc_dim)
        h, fbl = self.bilstm(x)                                # (N, T, 2H), (N, 2H)
        fe, fa = self.fwa(h, wc, fbl)                          # (N, 4H), (N, T)
        return fe, {"wc": wc, "fbl": fbl, "fa": fa}

    def apply_mcl_constraint(self) -> None:
        self.mcl.apply_mcl_constraint()


class ExtractorWithHead(nn.Module):
    """Phase-1 wrapper: extractor + temporary linear classifier."""

    def __init__(self, extractor: Extractor, n_classes: int):
        super().__init__()
        self.extractor = extractor
        self.head = nn.Linear(extractor.feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fe, _ = self.extractor(x)
        return self.head(fe), fe

    def apply_mcl_constraint(self) -> None:
        self.extractor.apply_mcl_constraint()
