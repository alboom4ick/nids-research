import torch

from nids_dl.layers import MCL


def test_mcl_shapes():
    m = MCL()
    x = torch.randn(8, 120)
    out = m(x)
    assert out.shape == (8, 56)


def test_mcl_custom_dims():
    m = MCL(in_features=64, n_filters=6, out_features=32)
    out = m(torch.randn(4, 64))
    assert out.shape == (4, 32)


def test_mcl_backward():
    m = MCL()
    x = torch.randn(2, 120, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


def test_mcl_constraint_sums_to_zero():
    """After apply_mcl_constraint each filter's weights must sum to zero."""
    m = MCL()
    m.apply_mcl_constraint()
    w = m.mcl_conv.weight.data                                 # (10, 1, 3)
    sums = w.sum(dim=-1)
    torch.testing.assert_close(sums, torch.zeros_like(sums), atol=1e-6, rtol=0)


def test_mcl_constraint_center_is_negative_sum_of_others():
    m = MCL()
    m.apply_mcl_constraint()
    w = m.mcl_conv.weight.data                                 # (10, 1, 3)
    center = w.shape[-1] // 2
    others_sum = w[..., [i for i in range(w.shape[-1]) if i != center]].sum(dim=-1)
    torch.testing.assert_close(w[..., center], -others_sum, atol=1e-6, rtol=0)


def test_mcl_constraint_is_idempotent():
    """Re-projecting twice should be a no-op (the set is closed)."""
    m = MCL()
    m.apply_mcl_constraint()
    w_once = m.mcl_conv.weight.data.clone()
    m.apply_mcl_constraint()
    torch.testing.assert_close(m.mcl_conv.weight.data, w_once, atol=1e-7, rtol=0)
