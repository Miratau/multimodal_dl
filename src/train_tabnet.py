import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

from src.data.ham10000 import (
    build_tabular_preprocessor,
    make_label_mapping,
    stratified_group_split,
)
from src.utils import set_seed

# TabNet configuration (fixed hyperparameters)
TABNET_CONFIG = {
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
    "tabnet": {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 5,
        "gamma": 1.5,
        "lambda_sparse": 1e-5,
        "max_epochs": 50,
        "patience": 10,
        "batch_size": 512,
        "virtual_batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "optimizer": "adam",
    },
}


def train_one_fold(
    df_tr,
    df_va,
    target_col,
    cfg,
):
    """Train and validate TabNet on a single fold."""
    tabnet_cfg = cfg["tabnet"]
    
    preproc = build_tabular_preprocessor(df_tr)
    X_tr = preproc.fit_transform(df_tr)
    X_va = preproc.transform(df_va)
    y_tr = df_tr[target_col].values
    y_va = df_va[target_col].values

    # Build per-sample weights to match TabNet's expectation
    classes = np.unique(y_tr)
    class_weights_vec = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_to_w = {c: w for c, w in zip(classes, class_weights_vec)}
    sample_weights = np.array([class_to_w[c] for c in y_tr], dtype=np.float32)

    # Create TabNet classifier
    clf = TabNetClassifier(
        n_d=tabnet_cfg["n_d"],
        n_a=tabnet_cfg["n_a"],
        n_steps=tabnet_cfg["n_steps"],
        gamma=tabnet_cfg["gamma"],
        lambda_sparse=tabnet_cfg["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=tabnet_cfg["lr"], weight_decay=tabnet_cfg["weight_decay"]),
        mask_type="sparsemax",
        verbose=0,
    )
    
    # Train with validation monitoring
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric=["balanced_accuracy", "logloss"],
        max_epochs=tabnet_cfg["max_epochs"],
        patience=tabnet_cfg["patience"],
        batch_size=tabnet_cfg["batch_size"],
        virtual_batch_size=tabnet_cfg["virtual_batch_size"],
        weights=sample_weights,
    )
    
    proba_va = clf.predict_proba(X_va).astype(np.float32)
    pred_va = proba_va.argmax(axis=1)
    
    return pred_va, proba_va


def main(config_path):
    """Train TabNet on HAM10000 dataset."""
    cfg = TABNET_CONFIG
    set_seed(cfg.get("seed", 2025))
    
    print("[INFO] Loading data...")
    data_csv = cfg["data"]["metadata_csv"]
    df = pd.read_csv(data_csv)
    
    # Encode labels
    label_map = make_label_mapping(df, target_col=cfg["data"]["target_col"])
    df = df.copy()
    df[cfg["data"]["target_col"]] = df[cfg["data"]["target_col"]].map(label_map).astype(int)
    
    print(f"[INFO] Found {len(label_map)} classes")
    
    out_dir = Path(cfg["logging"]["out_dir"]) / "tabnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-validation
    target_col = cfg["data"]["target_col"]
    group_col = cfg["data"].get("group_col", "lesion_id")
    folds = cfg["data"]["folds"]
    seed = cfg.get("seed", 2025)
    
    splits = stratified_group_split(df, n_splits=folds, target_col=target_col, group_col=group_col, seed=seed)
    oof_proba = np.zeros((len(df), len(label_map)), dtype=np.float32)
    
    print(f"[INFO] Running {folds}-fold cross-validation...")
    for fold, val_idx in enumerate(splits):
        print(f"\nFold {fold+1}/{folds}:")
        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[val_idx].copy()
        
        pred_va, proba_va = train_one_fold(df_tr, df_va, target_col, cfg)
        oof_proba[val_idx] = proba_va
        
        f1 = f1_score(df_va[target_col].values, pred_va, average="macro")
        print(f"  Fold F1: {f1:.4f}")
    
    np.save(out_dir / "oof_proba.npy", oof_proba)
    print(f"\n[OK] Saved TabNet OOF predictions to {out_dir / 'oof_proba.npy'}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Config path (unused, using TABNET_CONFIG)")
    args = parser.parse_args()
    raise SystemExit(main(args.config))

