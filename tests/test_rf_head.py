import numpy as np
import torch

from nids_dl import Extractor, RFConfig, evaluate, extract_features, fit_rf


def test_rf_fit_and_eval_smoke():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X = torch.randn(128, 120)
    y = torch.from_numpy(rng.integers(0, 2, size=128))
    extractor = Extractor()
    fe = extract_features(extractor, X, batch_size=32, device="cpu").numpy()
    clf = fit_rf(fe, y.numpy(), RFConfig(n_estimators=20, n_jobs=1, random_state=0))
    metrics = evaluate(clf, fe, y.numpy())
    for k in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics[k] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)
