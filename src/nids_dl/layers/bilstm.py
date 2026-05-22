import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Bidirectional LSTM consuming the raw 120-d feature row as a length-120
    sequence with 1 channel per step.

    Input  : (N, 120)       — preprocessed NSL-KDD row
    Output : hidden states  (N, 120, 2H)  — used as attention keys/values
             final state    (N, 2H)       — FBL, the "last bidirectional state"
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(-1)                            # (N, 120, 1)
        out, (h_n, _) = self.lstm(x)                       # out: (N, 120, 2H)
        # h_n: (2*num_layers, N, H). Concat forward/backward of the top layer.
        h_fwd = h_n[-2]                                    # (N, H)
        h_bwd = h_n[-1]                                    # (N, H)
        fbl = torch.cat([h_fwd, h_bwd], dim=-1)            # (N, 2H)
        return out, fbl
