"""Phase-2 classifier: scikit-learn RandomForest on DL features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class RFConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    class_weight: str | None = "balanced"
    random_state: int = 0
    extra: dict = field(default_factory=dict)


def fit_rf(Fe_train: np.ndarray, y_train: np.ndarray, cfg: RFConfig | None = None) -> RandomForestClassifier:
    cfg = cfg or RFConfig()
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        **cfg.extra,
    )
    clf.fit(Fe_train, y_train)
    return clf


def evaluate(clf: RandomForestClassifier, Fe: np.ndarray, y: np.ndarray) -> dict:
    y_pred = clf.predict(Fe)
    avg = "binary" if len(np.unique(y)) == 2 else "macro"
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y, y_pred, average=avg, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "report": classification_report(y, y_pred, zero_division=0),
    }
