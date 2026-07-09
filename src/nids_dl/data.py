"""NSL-KDD loader and preprocessor.

41 raw columns - 3 categorical (protocol_type, service, flag) - 2 near-constant
(num_outbound_cmds, is_host_login) + one-hot(3 + 70 + 11) = 120 features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
DROP_CONSTANT = ["num_outbound_cmds", "is_host_login"]

ATTACK_FAMILY = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
FAMILY_IDX = {"Normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4}


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X_train, X_test (N,120), y_bin_train/test, y_mul_train/test, feature names.

    Categorical encoding uses the union of train+test categories so both sets
    end up with identical column layouts.
    """
    train_df = train_df.drop(columns=["difficulty"]).copy()
    test_df = test_df.drop(columns=["difficulty"]).copy()

    for col in DROP_CONSTANT:
        train_df = train_df.drop(columns=col)
        test_df = test_df.drop(columns=col)

    y_bin_train = (train_df["label"] != "normal").astype(np.int64).to_numpy()
    y_bin_test = (test_df["label"] != "normal").astype(np.int64).to_numpy()

    def fam(lbl: str) -> int:
        return FAMILY_IDX[ATTACK_FAMILY.get(lbl, "R2L")]

    y_mul_train = train_df["label"].map(fam).to_numpy(dtype=np.int64)
    y_mul_test = test_df["label"].map(fam).to_numpy(dtype=np.int64)

    train_df = train_df.drop(columns="label")
    test_df = test_df.drop(columns="label")

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL, dtype=np.float32)

    feature_names = combined.columns.tolist()
    n_train = len(train_df)
    X_train = combined.iloc[:n_train].to_numpy(dtype=np.float32)
    X_test = combined.iloc[n_train:].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Reshape to (N, 10, 12) to match paper's 2D spatial mapping
    if X_train.shape[1] == 120:
        X_train = X_train.reshape(-1, 10, 12)
        X_test = X_test.reshape(-1, 10, 12)
    else:
        print(f"Warning: Expected 120 features, got {X_train.shape[1]}. Reshaping skipped.")

    return X_train, X_test, y_bin_train, y_bin_test, y_mul_train, y_mul_test, feature_names


def save_processed(
    out_dir: str | Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_bin_train: np.ndarray,
    y_bin_test: np.ndarray,
    y_mul_train: np.ndarray,
    y_mul_test: np.ndarray,
    feature_names: list[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": torch.from_numpy(X_train),
            "y_bin": torch.from_numpy(y_bin_train),
            "y_mul": torch.from_numpy(y_mul_train),
            "feature_names": feature_names,
        },
        out_dir / "train.pt",
    )
    torch.save(
        {
            "X": torch.from_numpy(X_test),
            "y_bin": torch.from_numpy(y_bin_test),
            "y_mul": torch.from_numpy(y_mul_test),
            "feature_names": feature_names,
        },
        out_dir / "test.pt",
    )


def load_processed(path: str | Path) -> dict:
    return torch.load(path, weights_only=False)


def build_and_save(
    train_path: str | Path,
    test_path: str | Path,
    out_dir: str | Path,
) -> dict:
    train_df = load_raw(train_path)
    test_df = load_raw(test_path)
    X_tr, X_te, yb_tr, yb_te, ym_tr, ym_te, feats = preprocess(train_df, test_df)
    save_processed(out_dir, X_tr, X_te, yb_tr, yb_te, ym_tr, ym_te, feats)
    return {
        "train_shape": X_tr.shape,
        "test_shape": X_te.shape,
        "n_features": len(feats),
        "feature_names": feats,
    }
