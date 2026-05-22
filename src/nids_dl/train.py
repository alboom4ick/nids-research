"""Phase-1 training loop for the DL feature extractor.

Uses a temporary softmax head with cross-entropy (paper §F says Adam updates
MCL weights; CE gives a differentiable label-based loss). After every Adam
step the MCL prediction-error-filter constraint is re-projected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .extractor import Extractor, ExtractorWithHead


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target: str = "binary"                                    # "binary" | "multi"
    log_every: int = 0                                        # 0 = silent
    seed: int = 0
    extractor_kwargs: dict = field(default_factory=dict)


def make_loader(X: torch.Tensor, y: torch.Tensor, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    ds = TensorDataset(X, y.long())
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=cfg.device.startswith("cuda")
    )


def train_extractor(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: TrainConfig | None = None,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> tuple[Extractor, list[dict]]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    n_classes = int(y_train.max().item()) + 1
    extractor = Extractor(in_features=X_train.shape[1], **cfg.extractor_kwargs)
    model = ExtractorWithHead(extractor, n_classes=n_classes).to(cfg.device)

    if cfg.device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, cfg, shuffle=True)
    val_loader = (
        make_loader(X_val, y_val, cfg, shuffle=False)
        if X_val is not None and y_val is not None
        else None
    )

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            underlying = model.module if isinstance(model, torch.nn.DataParallel) else model
            underlying.apply_mcl_constraint()

            bs = yb.size(0)
            running += loss.item() * bs
            seen += bs
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                print(f"epoch {epoch} step {step+1} loss {running/seen:.4f}")

        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "train_acc": correct / seen,
        }
        if val_loader is not None:
            row.update(_evaluate(model, val_loader, loss_fn, cfg.device))
        history.append(row)
        if cfg.log_every:
            print(row)

    return extractor, history


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str) -> dict:
    model.eval()
    running, seen, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits, _ = model(xb)
        loss = loss_fn(logits, yb)
        bs = yb.size(0)
        running += loss.item() * bs
        seen += bs
        correct += (logits.argmax(dim=-1) == yb).sum().item()
    return {"val_loss": running / seen, "val_acc": correct / seen}


@torch.no_grad()
def extract_features(
    extractor: Extractor,
    X: torch.Tensor,
    batch_size: int = 512,
    device: str | None = None,
) -> torch.Tensor:
    device = device or next(extractor.parameters()).device.type
    extractor.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size].to(device, non_blocking=True)
        fe, _ = extractor(xb)
        out.append(fe.cpu())
    return torch.cat(out, dim=0)
