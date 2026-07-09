import torch
from nids_dl.layers import FWA

def test_fwa_shapes():
    N, T, H2, Wc = 8, 10, 128, 56
    h = torch.randn(N, T, H2)
    wc = torch.randn(N, Wc)
    fbl = torch.randn(N, H2)
    m = FWA(hidden_dim=H2, wc_dim=Wc)
    fe, fa = m(h, wc, fbl)
    assert fe.shape == (N, 2 * H2)
    assert fa.shape == (N, T)

def test_fwa_weights_sum_to_one():
    m = FWA(hidden_dim=64, wc_dim=56)
    h = torch.randn(4, 10, 64)
    wc = torch.randn(4, 56)
    fbl = torch.randn(4, 64)
    _, fa = m(h, wc, fbl)
    torch.testing.assert_close(fa.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)

def test_fwa_depends_on_wc():
    """Changing Wc must change the attention distribution."""
    torch.manual_seed(0)
    m = FWA(hidden_dim=32, wc_dim=16)
    h = torch.randn(2, 10, 32)
    fbl = torch.randn(2, 32)
    wc_a = torch.randn(2, 16)
    wc_b = torch.randn(2, 16)
    _, fa_a = m(h, wc_a, fbl)
    _, fa_b = m(h, wc_b, fbl)
    assert not torch.allclose(fa_a, fa_b)

def test_fwa_backward():
    m = FWA(hidden_dim=32, wc_dim=16)
    h = torch.randn(2, 10, 32, requires_grad=True)
    wc = torch.randn(2, 16, requires_grad=True)
    fbl = torch.randn(2, 32)
    fe, _ = m(h, wc, fbl)
    fe.sum().backward()
    assert h.grad is not None
    assert wc.grad is not None
