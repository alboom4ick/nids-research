"""Phase-2 classifier: scikit-learn RandomForest and XGBoost on DL features."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

@dataclass
class TreeConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    random_state: int = 0
    extra: dict = field(default_factory=dict)

def fit_rf(Fe_train: np.ndarray, y_train: np.ndarray, cfg: TreeConfig | None = None) -> RandomForestClassifier:
    cfg = cfg or TreeConfig()
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
        class_weight="balanced",
        random_state=cfg.random_state,
        **cfg.extra,
    )
    clf.fit(Fe_train, y_train)
    return clf

def fit_xgb(Fe_train: np.ndarray, y_train: np.ndarray, cfg: TreeConfig | None = None) -> XGBClassifier:
    cfg = cfg or TreeConfig()
    clf = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        eval_metric='mlogloss',
        **cfg.extra,
    )
    clf.fit(Fe_train, y_train)
    return clf

def evaluate(clf, Fe: np.ndarray, y: np.ndarray) -> dict:
    y_pred = clf.predict(Fe)
    avg = "binary" if len(np.unique(y)) == 2 else "macro"
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y, y_pred, average=avg, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "report": classification_report(y, y_pred, zero_division=0),
    }
