"""CNN-MCL block (paper §F, Algorithm 1, Eqs 1-4).

Input is reshaped tabular data: (N, 10, 12).
Output is the Wc feature vector: (N, 56).
"""
from __future__ import annotations

import torch
from torch import nn

class MCL(nn.Module):
    def __init__(self, in_features: int = 120, n_filters: int = 10, out_features: int = 56) -> None:
        super().__init__()
        # Input is (N, 1, 10, 12)
        self.mcl_conv = nn.Conv2d(1, n_filters, kernel_size=3, padding=1, bias=False)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(n_filters, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (N, 32, 5, 6)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 16 * 5 * 6
        self.project = nn.Linear(self.flatten_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 10, 12) -> (N, 1, 10, 12)
        h = x.unsqueeze(1)
        h = torch.tanh(self.mcl_conv(h))                   # (N, 10, 10, 12)
        h = self.conv_stack(h)                             # (N, 16, 5, 6)
        h = h.flatten(1)                                   # (N, 480)
        return self.project(h)                             # (N, 56)

    @torch.no_grad()
    def apply_mcl_constraint(self) -> None:
        w = self.mcl_conv.weight.data                      # (10, 1, 3, 3)
        w_flat = w.view(w.shape[0], w.shape[1], -1)        # (10, 1, 9)
        k = w_flat.shape[-1]
        center = k // 2
        idx = [i for i in range(k) if i != center]
        idx_t = torch.tensor(idx, device=w.device)

        others = w_flat.index_select(-1, idx_t)
        denom = others.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = others / denom

        w_flat.index_copy_(-1, idx_t, normalized)
        w_flat[..., center] = -normalized.sum(dim=-1)
        w.copy_(w_flat.view(w.shape))
