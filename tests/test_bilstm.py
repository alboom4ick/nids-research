import torch
from nids_dl.layers.custom_rnn import CustomBiLSTM

def test_bilstm_shapes():
    m = CustomBiLSTM(input_size=12, hidden_size=64, wc_size=56)
    x = torch.randn(8, 10, 12)
    wc = torch.randn(8, 56)
    out, fbl = m(x, wc)
    assert out.shape == (8, 10, 128)
    assert fbl.shape == (8, 128)

def test_bilstm_backward():
    m = CustomBiLSTM(input_size=12, hidden_size=64, wc_size=56)
    x = torch.randn(2, 10, 12, requires_grad=True)
    wc = torch.randn(2, 56, requires_grad=True)
    _, fbl = m(x, wc)
    fbl.sum().backward()
    assert x.grad is not None
