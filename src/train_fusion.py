import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion_mlp import FusionMLP
from src.utils import set_seed

# Fusion model configuration (fixed hyperparameters)
FUSION_CONFIG = {
    "seed": 42,
    "device": "cuda",
    "num_workers": 4,
    "data": {
        "raw_dir": "data/raw/ham10000",
        "processed_dir": "data/processed",
        "metadata_csv": "data/processed/metadata.csv",
        "image_col": "image_path",
        "target_col": "dx",
        "group_col": "lesion_id",
        "folds": 5,
        "test_size": 0.2,
    },
    "logging": {
        "out_dir": "outputs",
        "tensorboard_dir": "outputs/tensorboard",
        "save_study_dir": "outputs/studies",
    },
    "fusion": {
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": 30,
        "patience": 5,
        "batch_size": 256,
    },
}


def train_fusion_fold(params, X_tr, y_tr, X_va, y_va, num_classes):
    """Train fusion MLP on a single fold and return validation F1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = FusionMLP(
        input_dim=X_tr.shape[1],
        num_classes=num_classes,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloaders
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))), batch_size=256, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.int64))), batch_size=256, shuffle=False)
    
    # Training loop
    best_f1 = -1.0
    epochs_no_improve = 0
    
    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        
        f1 = f1_score(np.concatenate(all_true), np.concatenate(all_pred), average="macro")
        
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= params["patience"]:
            break
    
    return best_f1


def main(config_path):
    """Train fusion MLP on OOF predictions from TabNet and ViT."""
    cfg = FUSION_CONFIG
    set_seed(cfg.get("seed", 2025))
    
    print("[INFO] Loading OOF predictions...")
    out_dir = Path(cfg["logging"]["out_dir"]) / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OOF predictions from both models
    tabnet_oof = Path(cfg["logging"]["out_dir"]) / "tabnet" / "oof_proba.npy"
    vit_oof = Path(cfg["logging"]["out_dir"]) / "vit" / "oof_proba.npy"
    
    if not tabnet_oof.exists() or not vit_oof.exists():
        raise FileNotFoundError("OOF files not found. Run train-tabnet and train-vit first!")
    
    X_tab = np.load(tabnet_oof)
    X_vit = np.load(vit_oof)
    X = np.concatenate([X_tab, X_vit], axis=1).astype(np.float32)
    
    # Load labels
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    y = df[cfg["data"]["target_col"]].astype("category").cat.codes.astype(np.int64).values
    num_classes = int(df[cfg["data"]["target_col"]].nunique())
    
    print(f"[INFO] X shape: {X.shape}, num_classes: {num_classes}")
    
    # Simple 5-fold CV to test fusion quality
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.get("seed", 2025))
    fusion_cfg = cfg["fusion"]
    scores = []
    
    print(f"[INFO] Running 5-fold CV with fusion MLP...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/5:", end=" ")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        f1 = train_fusion_fold(fusion_cfg, X_tr, y_tr, X_va, y_va, num_classes)
        scores.append(f1)
        print(f"F1: {f1:.4f}")
    
    print(f"\n[OK] Average F1 Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Config path (unused, using FUSION_CONFIG)")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
