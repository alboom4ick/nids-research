from torch._export.passes import replace_autocast_with_hop_pass
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nids_dl.data import build_and_save, load_processed
from nids_dl.train import train_extractor, TrainConfig, extract_features
from nids_dl.rf_head import fit_rf, fit_xgb, evaluate, TreeConfig

def main():
    print("Running end-to-end pipeline...")
    # Paths
    base_dir = Path(r"c:\Users\sduai\Documents\projis\research\nids_dl\nids-research")
    train_path = base_dir / "NSL-KDD" / "KDDTrain+.txt"
    test_path = base_dir / "NSL-KDD" / "KDDTest+.txt"
    out_dir = base_dir / "data" / "processed"
    
    # Check if data exists, if not, build it
    train_pt = out_dir / "train.pt"
    if not train_pt.exists():
        print("Preprocessing data...")
        build_and_save(train_path, test_path, out_dir)
        
    print("Loading preprocessed data...")
    train_data = load_processed(out_dir / "train.pt")
    test_data = load_processed(out_dir / "test.pt")
    
    X_tr, y_bin_tr = train_data["X"], train_data["y_bin"]
    X_te, y_bin_te = test_data["X"], test_data["y_bin"]
    
    print(f"Data shape: {X_tr.shape}")
    
    epoch_n = 100
    # Phase 1: Train Neural Network
    print(f"Phase 1: Training Neural Network Extractor... {epoch_n} epochs")
    cfg = TrainConfig(epochs=epoch_n, batch_size=256, device="cuda" if torch.cuda.is_available() else "cpu", log_every=10, target_val_acc=0.90, early_stopping_patience=20)
    extractor, hist = train_extractor(X_tr, y_bin_tr, cfg, X_val=X_te, y_val=y_bin_te)
    
    print("Extracting features...")
    fe_tr = extract_features(extractor, X_tr, device=cfg.device).numpy()
    fe_te = extract_features(extractor, X_te, device=cfg.device).numpy()
    
    # Phase 2: Train Classifiers
    print("Phase 2: Training Classifiers...")
    print("Training Random Forest...")
    rf_clf = fit_rf(fe_tr, y_bin_tr.numpy(), TreeConfig(n_estimators=100))
    rf_metrics = evaluate(rf_clf, fe_te, y_bin_te.numpy())
    
    print("Training XGBoost...")
    xgb_clf = fit_xgb(fe_tr, y_bin_tr.numpy(), TreeConfig(n_estimators=100))
    xgb_metrics = evaluate(xgb_clf, fe_te, y_bin_te.numpy())
    
    print("\n--- Results ---")
    print(f"Random Forest -> Accuracy: {rf_metrics['accuracy']:.4f}, Precision: {rf_metrics['precision']:.4f}, Recall: {rf_metrics['recall']:.4f}, F1: {rf_metrics['f1']:.4f}")
    print(f"XGBoost       -> Accuracy: {xgb_metrics['accuracy']:.4f}, Precision: {xgb_metrics['precision']:.4f}, Recall: {xgb_metrics['recall']:.4f}, F1: {xgb_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
