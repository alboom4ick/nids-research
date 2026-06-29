from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_raw_cic(data_dir: str | Path) -> pd.DataFrame:
    """Load all CSVs from the CIS-IDS-2017 directory and merge them."""
    data_dir = Path(data_dir)
    csv_files = glob.glob(str(data_dir / "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
        
    print(f"Found {len(csv_files)} CSV files. Loading data (this might take a minute or two)...")
    
    dfs = []
    for file in csv_files:
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        dfs.append(df)
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Strip whitespace from columns
    merged_df.columns = merged_df.columns.str.strip()
    return merged_df


def preprocess_cic(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X_train, X_test (N, 10, 12), y_bin_train, y_bin_test, feature names."""
    print(f"Initial shape: {df.shape}")
    
    # Sample to 160,000 rows to yield exactly 128,000 train samples (500 steps at batch_size=256)
    target_size = 160000
    if len(df) > target_size:
        print(f"Sampling {target_size} rows from {len(df)} total rows...")
        df = df.sample(n=target_size, random_state=42).reset_index(drop=True)
    
    # Drop duplicate columns (Fwd Header Length usually appears twice)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Replace Infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaNs with 0
    df.fillna(0, inplace=True)
    
    # Separate Label
    label_col = "Label" if "Label" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"Could not find label column. Columns: {df.columns.tolist()}")
        
    y_bin = (df[label_col] != "BENIGN").astype(np.int64).to_numpy()
    
    df = df.drop(columns=[label_col])
    feature_names = df.columns.tolist()
    
    # Convert to numpy
    X = df.to_numpy(dtype=np.float32)
    print(f"Features extracted shape: {X.shape}")
    
    # Split train/test (80/20) FIRST to prevent data leakage
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    # Scale continuous features using ONLY the training set
    print("Scaling continuous features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    # Zero pad to 120 features
    n_features = X_train.shape[1]
    if n_features < 120:
        padding = 120 - n_features
        print(f"Padding {padding} zeros to reach 120 features...")
        X_train = np.pad(X_train, ((0, 0), (0, padding)), 'constant', constant_values=0)
        X_test = np.pad(X_test, ((0, 0), (0, padding)), 'constant', constant_values=0)
        feature_names += [f"padding_{i}" for i in range(padding)]
    elif n_features > 120:
        print(f"Warning: Extracted {n_features} features, which is > 120. Truncating to 120.")
        X_train = X_train[:, :120]
        X_test = X_test[:, :120]
        feature_names = feature_names[:120]
        
    # Reshape to (N, 10, 12)
    X_train = X_train.reshape(-1, 10, 12)
    X_test = X_test.reshape(-1, 10, 12)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


def save_processed_cic(
    out_dir: str | Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_bin_train: np.ndarray,
    y_bin_test: np.ndarray,
    feature_names: list[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We create fake y_mul since the model expects it, but we only train on binary for now.
    y_mul_train = np.zeros_like(y_bin_train)
    y_mul_test = np.zeros_like(y_bin_test)
    
    print(f"Saving processed data to {out_dir}...")
    torch.save(
        {
            "X": torch.from_numpy(X_train),
            "y_bin": torch.from_numpy(y_bin_train),
            "y_mul": torch.from_numpy(y_mul_train),
            "feature_names": feature_names,
        },
        out_dir / "train_cic.pt",
    )
    torch.save(
        {
            "X": torch.from_numpy(X_test),
            "y_bin": torch.from_numpy(y_bin_test),
            "y_mul": torch.from_numpy(y_mul_test),
            "feature_names": feature_names,
        },
        out_dir / "test_cic.pt",
    )
    print("Saved successfully!")


def build_and_save_cic(
    data_dir: str | Path,
    out_dir: str | Path,
) -> dict:
    df = load_raw_cic(data_dir)
    X_tr, X_te, yb_tr, yb_te, feats = preprocess_cic(df)
    save_processed_cic(out_dir, X_tr, X_te, yb_tr, yb_te, feats)
    return {
        "train_shape": X_tr.shape,
        "test_shape": X_te.shape,
        "n_features": len(feats),
        "feature_names": feats,
    }
