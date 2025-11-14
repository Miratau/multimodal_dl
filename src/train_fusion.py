import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion_mlp import FusionMLP
from src.utils import set_seed

CONFIG = {
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
        "val_size": 0.1,
        "test_size": 0.1,
    },
    "logging": {
        "out_dir": "artifacts",
        "tensorboard_dir": "artifacts/tensorboard",
        "save_study_dir": "artifacts/studies",
    },
    "fusion": {
        "epochs": 30,
        "patience": 7,
        "batch_size": 256,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "lr": 0.001,
    },
}


def _load_features(base_dir):
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

    train_feats = np.concatenate([tab_train, vit_train], axis=1).astype(np.float32)
    val_feats = np.concatenate([tab_val, vit_val], axis=1).astype(np.float32)
    test_feats = np.concatenate([tab_test, vit_test], axis=1).astype(np.float32)
    return train_feats, val_feats, test_feats, train_labels, val_labels, test_labels


def _build_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _train_fusion(train_loader, val_loader, input_dim, num_classes, cfg):
    fusion_cfg = cfg["fusion"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = FusionMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=fusion_cfg["hidden_dim"],
        num_layers=fusion_cfg["num_layers"],
        dropout=fusion_cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fusion_cfg["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_preds = None
    best_proba = None
    no_improve = 0

    for epoch in range(fusion_cfg["epochs"]):
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

        model.eval()
        all_proba = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)
                all_true.append(yb.numpy())
        proba_val = np.concatenate(all_proba)
        y_val = np.concatenate(all_true)
        preds = proba_val.argmax(axis=1)
        macro_f1 = f1_score(y_val, preds, average="macro")
        print(f"[Fusion] Epoch {epoch + 1}/{fusion_cfg['epochs']} | Train loss: {epoch_loss:.4f} | Val macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = model.state_dict()
            best_preds = preds
            best_proba = proba_val
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= fusion_cfg["patience"]:
            print("[Fusion] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_proba is None or best_preds is None:
        model.eval()
        all_proba = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)
                all_true.append(yb.numpy())
        best_proba = np.concatenate(all_proba)
        y_val = np.concatenate(all_true)
        best_preds = best_proba.argmax(axis=1)
        best_f1 = f1_score(y_val, best_preds, average="macro")

    metrics = {"macro_f1": float(best_f1)}
    return model, metrics, best_proba, best_preds


def run_training(cfg, out_dir, tune):
    if tune:
        print("[WARN] Tuning disabled; proceeding with fixed Fusion parameters.")
    set_seed(cfg.get("seed", 42))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(cfg["logging"]["out_dir"])
    X_train, X_val, X_test, y_train, y_val, y_test = _load_features(base_dir)
    train_loader, val_loader = _build_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=cfg["fusion"]["batch_size"],
    )
    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    model, metrics, proba_val, preds_val = _train_fusion(
        train_loader,
        val_loader,
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        cfg=cfg,
    )

    model.eval()
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=cfg["fusion"]["batch_size"], shuffle=False)
    all_proba = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
            all_true.append(yb.numpy())
    proba_test = np.concatenate(all_proba)
    y_test_np = np.concatenate(all_true)
    preds_test = proba_test.argmax(axis=1)
    metrics["macro_f1_test"] = float(f1_score(y_test_np, preds_test, average="macro"))

    print(f"\n[Fusion] Training completed!")
    print("[Fusion] Test classification report:")
    print(classification_report(y_test_np, preds_test))

    torch.save(model.state_dict(), out_dir / "model_best.pth")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(out_dir / "val_proba.npy", proba_val.astype(np.float32))
    np.save(out_dir / "val_preds.npy", preds_val.astype(np.int64))
    np.save(out_dir / "val_labels.npy", y_val.astype(np.int64))
    np.save(out_dir / "test_proba.npy", proba_test.astype(np.float32))
    np.save(out_dir / "test_preds.npy", preds_test.astype(np.int64))
    np.save(out_dir / "test_labels.npy", y_test_np.astype(np.int64))


def main(tune):
    out_dir = Path(CONFIG["logging"]["out_dir"]) / "fusion"
    run_training(CONFIG, out_dir, tune=tune)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.tune))
