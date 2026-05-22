import torch

from nids_dl.layers import BiLSTM


def test_bilstm_shapes():
    m = BiLSTM(hidden_size=64)
    x = torch.randn(8, 120)
    out, fbl = m(x)
    assert out.shape == (8, 120, 128)
    assert fbl.shape == (8, 128)


def test_bilstm_accepts_3d_input():
    m = BiLSTM(hidden_size=32)
    x = torch.randn(4, 120, 1)
    out, fbl = m(x)
    assert out.shape == (4, 120, 64)
    assert fbl.shape == (4, 64)


def test_bilstm_backward():
    m = BiLSTM(hidden_size=16)
    x = torch.randn(2, 120, requires_grad=True)
    _, fbl = m(x)
    fbl.sum().backward()
    assert x.grad is not None
