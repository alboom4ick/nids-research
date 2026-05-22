"""CNN-MCL block (paper §F, Algorithm 1, Eqs 1–4).

Shape flow per plan.md:
    (N, 120) -> (N, 10, 120)  [MCL]
             -> (N, 56)        [Conv2d stack + flatten]

The Mean-Convolutional constraint (Eqs 3–4) is enforced on the 10 filter
weights of the first Conv1d after every Adam step: the central weight of
each filter is set to the negative mean of its other weights, and the
remaining weights are normalized by the sum of filter weights so that each
filter's weights sum to zero. This is the prediction-error-filter property
the paper inherits from steganalysis.
"""
from __future__ import annotations

import torch
from torch import nn


class MCL(nn.Module):
    """CNN-MCL block: (N, 120) -> (N, 56)."""

    def __init__(self, in_features: int = 120, n_filters: int = 10,
                 out_features: int = 56) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_filters = n_filters

        # Eq 1: per-filter 1-d conv over the feature vector. kernel_size=3 with
        # padding=1 preserves length so each filter produces a length-120 map.
        # The central weight is index 1; Eqs 3–4 constrain it to -mean(others).
        self.mcl_conv = nn.Conv1d(
            in_channels=1, out_channels=n_filters,
            kernel_size=3, padding=1, bias=False,
        )

        # Conv2d stack on (N, 1, n_filters, in_features) = (N, 1, 10, 120).
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # -> (N, 32, 5, 60)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                  # -> (N, 16, 1, 1)
        )
        self.project = nn.Linear(16, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 120) -> (N, 1, 120)
        h = x.unsqueeze(1)
        # Eq 1: filtered maps, then tanh per Eq 3.
        h = torch.tanh(self.mcl_conv(h))                   # (N, 10, 120)
        # (N, 1, 10, 120) for 2-d stack
        h = h.unsqueeze(1)
        h = self.conv_stack(h)                             # (N, 16, 1, 1)
        h = h.flatten(1)                                   # (N, 16)
        return self.project(h)                             # (N, 56)

    @torch.no_grad()
    def apply_mcl_constraint(self) -> None:
        """Re-project `mcl_conv.weight` onto the prediction-error-filter set.

        Call after every optimizer step (paper §F, Algorithm 1 Step 4). The
        paper's Eqs 3–4 are literally ambiguous; the invariant the filter
        family was designed to satisfy (paper inherits the construction from
        steganalysis SRM filters) is that each filter's weights sum to zero,
        with the central tap carrying the negative of the surrounding sum.

        Weight shape: (n_filters, 1, kernel_size). For each filter we
        normalize the non-central weights to unit L1 norm, then set the
        central weight to -sum(others) so the filter sums to zero.
        """
        w = self.mcl_conv.weight.data                      # (10, 1, 3)
        k = w.shape[-1]
        center = k // 2
        idx = [i for i in range(k) if i != center]
        idx_t = torch.tensor(idx, device=w.device)

        others = w.index_select(-1, idx_t)                 # (10, 1, k-1)
        denom = others.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = others / denom                        # L1-normalized per filter

        w.index_copy_(-1, idx_t, normalized)
        w[..., center] = -normalized.sum(dim=-1)
