import torch
import torch.nn as nn


class FWA(nn.Module):
    """Feature-Weighted Attention / BI-ATT (paper Algorithm 2, lines 395–405;
    Eqs 1–2 at lines 436–446).

    Implements Bahdanau-style additive attention where the CNN-MCL summary Wc
    (shape (N, wc_dim)) acts as the query that reweights the BI-LSTM hidden
    states h_1..h_T:

        score_t = v^T · tanh(W1 · h_t + W2 · Wc + b)
        FA_t    = softmax_t(score_t)
        FAT     = Σ_t FA_t · h_t
        Fe      = concat(FBL, FAT)

    This matches the paper's signature `Fe ← BI-ATT(Wc, F)` and the name
    "feature-weighted": Wc literally weights the attention distribution.
    """

    def __init__(self, hidden_dim: int, wc_dim: int, attn_dim: int | None = None):
        super().__init__()
        if attn_dim is None:
            attn_dim = hidden_dim
        self.W1 = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(wc_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,                                   # (N, T, 2H)
        wc: torch.Tensor,                                  # (N, wc_dim)
        fbl: torch.Tensor,                                 # (N, 2H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W2(wc).unsqueeze(1)                       # (N, 1, attn_dim)
        k = self.W1(h)                                     # (N, T, attn_dim)
        scores = self.v(torch.tanh(k + q)).squeeze(-1)     # (N, T)
        fa = torch.softmax(scores, dim=-1)                 # (N, T)
        fat = torch.bmm(fa.unsqueeze(1), h).squeeze(1)     # (N, 2H)
        fe = torch.cat([fbl, fat], dim=-1)                 # (N, 4H)
        return fe, fa
