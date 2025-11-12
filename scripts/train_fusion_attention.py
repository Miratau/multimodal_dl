"""
Training script for Attention-based Fusion Model
Combines TabNet and ViT predictions using learned attention weights
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion_mlp_attention import AttentionFusion
from src.utils.seed import set_seed


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    include = cfg.get("include")
    if include:
        with open(include, "r") as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def _load_features(base_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed TabNet and ViT probabilities"""
    tab_dir = base_dir / "tabnet"
    vit_dir = base_dir / "vit"
    
    required = [
        tab_dir / "train_proba.npy",
        tab_dir / "val_proba.npy",
        tab_dir / "test_proba.npy",
        tab_dir / "train_labels.npy",
        tab_dir / "val_labels.npy",
        tab_dir / "test_labels.npy",
        vit_dir / "train_proba.npy",
        vit_dir / "val_proba.npy",
        vit_dir / "test_proba.npy",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

    tab_train = np.load(tab_dir / "train_proba.npy")
    tab_val = np.load(tab_dir / "val_proba.npy")
    tab_test = np.load(tab_dir / "test_proba.npy")
    train_labels = np.load(tab_dir / "train_labels.npy").astype(np.int64)
    val_labels = np.load(tab_dir / "val_labels.npy").astype(np.int64)
    test_labels = np.load(tab_dir / "test_labels.npy").astype(np.int64)
    vit_train = np.load(vit_dir / "train_proba.npy")
    vit_val = np.load(vit_dir / "val_proba.npy")
    vit_test = np.load(vit_dir / "test_proba.npy")

    # Concatenate: [TabNet_proba, ViT_proba]
    train_feats = np.concatenate([tab_train, vit_train], axis=1).astype(np.float32)
    val_feats = np.concatenate([tab_val, vit_val], axis=1).astype(np.float32)
    test_feats = np.concatenate([tab_test, vit_test], axis=1).astype(np.float32)
    return train_feats, val_feats, test_feats, train_labels, val_labels, test_labels


def _build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _train_attention_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    cfg: Dict[str, Any],
) -> tuple[AttentionFusion, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    fusion_cfg = cfg["fusion_attention"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Create attention-based fusion model
    model = AttentionFusion(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=fusion_cfg["hidden_dim"],
        dropout=fusion_cfg["dropout"],
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=fusion_cfg["lr"], 
        weight_decay=fusion_cfg.get("weight_decay", 1e-4)
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_preds = None
    best_proba = None
    best_attention_weights = None
    no_improve = 0

    print(f"[Attention Fusion] Training on device: {device}")
    print(f"[Attention Fusion] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(fusion_cfg["epochs"]):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        all_proba = []
        all_true = []
        all_attention = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)
                all_true.append(yb.numpy())
                
                # Get attention weights for analysis
                attn = model.get_attention_weights(xb).cpu().numpy()
                all_attention.append(attn)
        
        proba_val = np.concatenate(all_proba)
        y_val = np.concatenate(all_true)
        attention_weights = np.concatenate(all_attention)
        preds = proba_val.argmax(axis=1)
        macro_f1 = f1_score(y_val, preds, average="macro")
        
        # Print epoch stats including average attention weights
        avg_tabnet_attn = attention_weights[:, 0].mean()
        avg_vit_attn = attention_weights[:, 1].mean()
        
        print(f"[Attention Fusion] Epoch {epoch + 1}/{fusion_cfg['epochs']} | "
              f"Train loss: {epoch_loss:.4f} | Val macro F1: {macro_f1:.4f} | "
              f"Avg attention [TabNet: {avg_tabnet_attn:.3f}, ViT: {avg_vit_attn:.3f}]")

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = model.state_dict()
            best_preds = preds
            best_proba = proba_val
            best_attention_weights = attention_weights
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= fusion_cfg["patience"]:
            print("[Attention Fusion] Early stopping triggered.")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # If no best found, compute final predictions
    if best_proba is None or best_preds is None:
        model.eval()
        all_proba = []
        all_true = []
        all_attention = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)
                all_true.append(yb.numpy())
                attn = model.get_attention_weights(xb).cpu().numpy()
                all_attention.append(attn)
        best_proba = np.concatenate(all_proba)
        y_val = np.concatenate(all_true)
        best_attention_weights = np.concatenate(all_attention)
        best_preds = best_proba.argmax(axis=1)
        best_f1 = f1_score(y_val, best_preds, average="macro")

    metrics = {"macro_f1": float(best_f1)}
    return model, metrics, best_proba, best_preds, best_attention_weights


def run_training(cfg: Dict[str, Any], out_dir: Path, tune: bool) -> None:
    if tune:
        print("[WARN] Tuning disabled; proceeding with fixed Attention Fusion parameters.")
    set_seed(cfg.get("seed", 2025))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(cfg["logging"]["out_dir"])
    X_train, X_val, X_test, y_train, y_val, y_test = _load_features(base_dir)
    
    print(f"[Attention Fusion] Train features shape: {X_train.shape}")
    print(f"[Attention Fusion] Val features shape: {X_val.shape}")
    
    train_loader, val_loader = _build_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=cfg["fusion_attention"]["batch_size"],
    )
    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    
    model, metrics, proba_val, preds_val, attention_weights = _train_attention_fusion(
        train_loader,
        val_loader,
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        cfg=cfg,
    )

    # Evaluate on test
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=cfg["fusion_attention"]["batch_size"], shuffle=False)
    model.eval()
    all_proba = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
    proba_test = np.concatenate(all_proba)
    preds_test = proba_test.argmax(axis=1)
    metrics["macro_f1_test"] = float(f1_score(y_test, preds_test, average="macro"))

    # Print final results
    print(f"\n[Attention Fusion] Training completed!")
    print("\n[Attention Fusion] Test classification report:")
    print(classification_report(y_test, preds_test))

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "model_best.pth")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(out_dir / "val_proba.npy", proba_val.astype(np.float32))
    np.save(out_dir / "val_preds.npy", preds_val.astype(np.int64))
    np.save(out_dir / "val_labels.npy", y_val.astype(np.int64))
    np.save(out_dir / "attention_weights.npy", attention_weights.astype(np.float32))
    np.save(out_dir / "test_proba.npy", proba_test.astype(np.float32))
    np.save(out_dir / "test_preds.npy", preds_test.astype(np.int64))
    np.save(out_dir / "test_labels.npy", y_test.astype(np.int64))
    
    print(f"\n[Attention Fusion] Artifacts saved to {out_dir}")
    print(f"  - Model checkpoint")
    print(f"  - Val/test predictions and labels")
    print(f"  - Attention weights (for analysis)")


def main(config_path: str, tune: bool) -> int:
    cfg = _load_config(config_path)
    out_dir = Path(cfg["logging"]["out_dir"]) / "fusion_attention"
    run_training(cfg, out_dir, tune=tune)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion_attention.yaml")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.tune))

