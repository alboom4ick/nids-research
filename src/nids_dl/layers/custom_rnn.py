"""Custom Bidirectional RNN cell fusing CNN features."""
from __future__ import annotations

import torch
import torch.nn as nn

class CustomRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, wc_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_c_proj = nn.Linear(wc_size, hidden_size, bias=False)
        
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, w_c: torch.Tensor) -> torch.Tensor:
        # F_B = tanh(W_c * F_t + U_c * F_t-1 + b_c) in paper, mapped to this projection
        return torch.tanh(self.W_x(x_t) + self.U_c(h_prev) + self.W_c_proj(w_c))

class CustomBiLSTM(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 64, wc_size: int = 56):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_fwd = CustomRNNCell(input_size, hidden_size, wc_size)
        self.cell_bwd = CustomRNNCell(input_size, hidden_size, wc_size)
        
    def forward(self, x: torch.Tensor, w_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, T, input_size) -> (N, 10, 12)
        N, T, _ = x.size()
        device = x.device
        
        h_fwd = torch.zeros(N, self.hidden_size, device=device)
        h_bwd = torch.zeros(N, self.hidden_size, device=device)
        
        out_fwd = []
        for t in range(T):
            h_fwd = self.cell_fwd(x[:, t, :], h_fwd, w_c)
            out_fwd.append(h_fwd)
            
        out_bwd = []
        for t in reversed(range(T)):
            h_bwd = self.cell_bwd(x[:, t, :], h_bwd, w_c)
            out_bwd.insert(0, h_bwd)
            
        out_fwd = torch.stack(out_fwd, dim=1) # (N, T, H)
        out_bwd = torch.stack(out_bwd, dim=1) # (N, T, H)
        
        out = torch.cat([out_fwd, out_bwd], dim=-1) # (N, T, 2H)
        fbl = torch.cat([out_fwd[:, -1, :], out_bwd[:, 0, :]], dim=-1) # (N, 2H)
        
        return out, fbl
