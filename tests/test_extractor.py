import torch

from nids_dl import Extractor, ExtractorWithHead, TrainConfig, train_extractor, extract_features


def test_extractor_shapes():
    m = Extractor()
    fe, aux = m(torch.randn(8, 120))
    assert fe.shape == (8, 4 * 64)
    assert aux["wc"].shape == (8, 56)
    assert aux["fbl"].shape == (8, 2 * 64)
    assert aux["fa"].shape == (8, 120)


def test_extractor_backward_and_mcl_constraint():
    m = Extractor()
    fe, _ = m(torch.randn(4, 120))
    fe.sum().backward()
    m.apply_mcl_constraint()
    w = m.mcl.mcl_conv.weight.data
    torch.testing.assert_close(w.sum(dim=-1), torch.zeros(w.shape[:-1]), atol=1e-6, rtol=0)


def test_extractor_with_head_forward():
    m = ExtractorWithHead(Extractor(), n_classes=2)
    logits, fe = m(torch.randn(3, 120))
    assert logits.shape == (3, 2)
    assert fe.shape == (3, 256)


def test_train_extractor_smoke():
    torch.manual_seed(0)
    X = torch.randn(64, 120)
    y = torch.randint(0, 2, (64,))
    cfg = TrainConfig(epochs=1, batch_size=16, device="cpu", seed=0)
    extractor, hist = train_extractor(X, y, cfg)
    assert len(hist) == 1 and "train_loss" in hist[0]
    fe = extract_features(extractor, X, batch_size=32, device="cpu")
    assert fe.shape == (64, 256)
    # MCL constraint is maintained after training.
    w = extractor.mcl.mcl_conv.weight.data
    torch.testing.assert_close(w.sum(dim=-1), torch.zeros(w.shape[:-1]), atol=1e-5, rtol=0)
