import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.ham10000 import Ham10000Dataset, stratified_group_split
from src.models.vit_module import create_vit, split_parameters
from src.utils import set_seed, build_train_transforms, build_val_transforms, train_epoch, validate

# Vision Transformer configuration (fixed hyperparameters)
VIT_CONFIG = {
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
    "vit": {
        "backbone": "vit_small_patch16_224",
        "img_size": 224,
        "epochs": 30,
        "patience": 5,
        "batch_size": 32,
        "label_smoothing": 0.1,
        "lr_head": 5e-4,
        "lr_backbone_mult": 0.25,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "amp": True,
    },
}


def train_validate_fold(
    df_tr,
    df_va,
    cfg,
    device,
    num_classes,
):
    """Train and validate ViT on a single fold."""
    vit_cfg = cfg["vit"]
    
    train_tf = build_train_transforms(vit_cfg["img_size"])
    val_tf = build_val_transforms(vit_cfg["img_size"])
    
    identity_map = {i: i for i in range(num_classes)}
    ds_tr = Ham10000Dataset(df_tr, class_to_idx=identity_map, image_transform=train_tf, use_tabular=False)
    ds_va = Ham10000Dataset(df_va, class_to_idx=identity_map, image_transform=val_tf, use_tabular=False)
    
    dl_tr = DataLoader(ds_tr, batch_size=vit_cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=vit_cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = create_vit(vit_cfg["backbone"], num_classes=num_classes, pretrained=True).to(device)
    
    # Optimizer with differential learning rates
    backbone_params, head_params = split_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": vit_cfg["lr_head"] * vit_cfg["lr_backbone_mult"]},
            {"params": head_params, "lr": vit_cfg["lr_head"]},
        ],
        weight_decay=vit_cfg["weight_decay"],
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=vit_cfg["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=vit_cfg["amp"])
    
    # Training loop with early stopping
    best_f1 = -1.0
    best_proba = None
    epochs_no_improve = 0
    
    for epoch in range(vit_cfg["epochs"]):
        train_epoch(model, dl_tr, criterion, optimizer, device, scaler, use_amp=vit_cfg["amp"])
        val_loss, proba_va, y_true = validate(model, dl_va, criterion, device)
        pred = proba_va.argmax(axis=1)
        f1 = f1_score(y_true, pred, average="macro")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{vit_cfg['epochs']} - F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_proba = proba_va.copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= vit_cfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    return best_proba


def main(config_path):
    """Train ViT on HAM10000 dataset."""
    cfg = VIT_CONFIG
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 2025))
    
    print("[INFO] Loading data...")
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    
    # Encode labels to integers 0..C-1
    label_map = {c: i for i, c in enumerate(sorted(df[cfg["data"]["target_col"]].astype(str).unique()))}
    df[cfg["data"]["target_col"]] = df[cfg["data"]["target_col"]].map(label_map).astype(int)
    num_classes = len(label_map)
    
    print(f"[INFO] Found {num_classes} classes")
    
    out_dir = Path(cfg["logging"]["out_dir"]) / "vit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Cross-validation with OOF predictions
    target_col = cfg["data"]["target_col"]
    group_col = cfg["data"].get("group_col", "lesion_id")
    folds = cfg["data"]["folds"]
    seed = cfg.get("seed", 2025)
    
    splits = stratified_group_split(df, n_splits=folds, target_col=target_col, group_col=group_col, seed=seed)
    oof_proba = np.zeros((len(df), num_classes), dtype=np.float32)
    
    print(f"[INFO] Running {folds}-fold cross-validation...")
    for fold, val_idx in enumerate(splits):
        print(f"\nFold {fold+1}/{folds}:")
        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[val_idx].copy()
        
        proba_va = train_validate_fold(df_tr, df_va, cfg, device, num_classes)
        oof_proba[val_idx] = proba_va
    
    np.save(out_dir / "oof_proba.npy", oof_proba)
    print(f"\n[OK] Saved ViT OOF predictions to {out_dir / 'oof_proba.npy'}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Config path (unused, using VIT_CONFIG)")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
