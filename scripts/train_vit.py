import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.data.ham10000 import Ham10000Dataset, make_label_mapping
from src.models.vit_module import create_vit, split_parameters
from src.training.engine import train_one_epoch, validate
from src.utils.seed import set_seed
from src.utils.transforms import build_train_transforms, build_val_transforms
from scripts.utils import get_train_val_test_metadata


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


def _prepare_metadata(train_df, val_df, target_col: str) -> tuple:
    label_map = make_label_mapping(train_df, target_col=target_col)
    num_classes = len(label_map)

    train_enc = train_df.copy()
    val_enc = val_df.copy()
    train_enc[target_col] = train_enc[target_col].map(label_map).astype(int)
    val_enc[target_col] = val_enc[target_col].map(label_map).astype(int)

    return train_enc, val_enc, label_map, num_classes


def _build_dataloaders(train_df, val_df, cfg: Dict[str, Any]) -> tuple:
    img_size = cfg["vit"]["img_size"]
    batch_size = cfg["vit"]["batch_size"]
    num_workers = cfg.get("num_workers", 4)
    target_col = cfg["data"]["target_col"]

    train_tf = build_train_transforms(img_size)
    val_tf = build_val_transforms(img_size)

    num_classes = len(sorted(train_df[target_col].unique()))
    identity_map = {i: i for i in range(num_classes)}

    train_dataset = Ham10000Dataset(train_df, class_to_idx=identity_map, image_transform=train_tf, use_tabular=False, target_col=target_col)
    val_dataset = Ham10000Dataset(val_df, class_to_idx=identity_map, image_transform=val_tf, use_tabular=False, target_col=target_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, train_eval_loader, val_loader


def _train_vit(train_loader, val_loader, cfg: Dict[str, Any], num_classes: int, device: torch.device):
    vit_cfg = cfg["vit"]
    model = create_vit(vit_cfg["backbone"], num_classes=num_classes, pretrained=True).to(device)
    backbone_params, head_params = split_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": vit_cfg["lr_head"] * vit_cfg["lr_backbone_mult"]},
            {"params": head_params, "lr": vit_cfg["lr_head"]},
        ],
        weight_decay=vit_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=vit_cfg["label_smoothing"])
    scaler = torch.amp.GradScaler('cuda', enabled=vit_cfg.get("amp", True))

    best_state = None
    best_metrics = {"macro_f1": -1.0, "val_loss": float("inf")}
    best_proba = None
    no_improve = 0

    for epoch in range(vit_cfg["epochs"]):
        print(f"[ViT] Epoch {epoch + 1}/{vit_cfg['epochs']}")
        train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp=vit_cfg.get("amp", True))
        val_loss, proba_val, y_true = validate(model, val_loader, criterion, device)
        preds = proba_val.argmax(axis=1)
        macro_f1 = f1_score(y_true, preds, average="macro")
        print(f"[ViT] Val loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f}")

        improved = macro_f1 > best_metrics["macro_f1"]
        if improved:
            best_metrics.update({"macro_f1": macro_f1, "val_loss": val_loss})
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            best_proba = proba_val.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= vit_cfg["patience"]:
            print("[ViT] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
    return model, best_proba, best_metrics


def _collect_logits(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    _, proba, y_true = validate(model, loader, criterion, device)
    return proba, y_true


def run_training(cfg: Dict[str, Any], out_dir: Path, tune: bool) -> None:
    if tune:
        print("[WARN] Tuning disabled; proceeding with fixed ViT parameters.")
    set_seed(cfg.get("seed", 2025))
    out_dir.mkdir(parents=True, exist_ok=True)

    val_size = cfg["data"].get("val_size", 0.1)
    test_size = cfg["data"].get("test_size", 0.1)
    train_meta, val_meta, test_meta = get_train_val_test_metadata(
        val_size=val_size,
        test_size=test_size,
        seed=cfg.get("seed", 2025),
    )
    target_col = cfg["data"]["target_col"]
    train_enc, val_enc, label_map, num_classes = _prepare_metadata(train_meta, val_meta, target_col)
    # Encode test using train label_map
    test_enc = test_meta.copy()
    test_enc[target_col] = test_enc[target_col].map(label_map).astype(int)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, train_eval_loader, val_loader = _build_dataloaders(train_enc, val_enc, cfg)
    model, proba_val, metrics = _train_vit(train_loader, val_loader, cfg, num_classes, device)

    val_labels = val_enc[target_col].values.astype(np.int64)
    if proba_val is None:
        val_loss, proba_val, val_labels = validate(model, val_loader, nn.CrossEntropyLoss(), device)
        metrics["val_loss"] = float(val_loss)

    # Build test loader and evaluate on test set
    img_size = cfg["vit"]["img_size"]
    target = cfg["data"]["target_col"]
    val_tf = build_val_transforms(img_size)
    num_classes_eval = len(sorted(train_enc[target].unique()))
    identity_map = {i: i for i in range(num_classes_eval)}
    test_dataset = Ham10000Dataset(test_enc, class_to_idx=identity_map, image_transform=val_tf, use_tabular=False, target_col=target)
    test_loader = DataLoader(test_dataset, batch_size=cfg["vit"]["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)
    test_loss, proba_test, test_labels = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    test_preds = proba_test.argmax(axis=1)
    test_macro = f1_score(test_labels, test_preds, average="macro")
    metrics["macro_f1_test"] = float(test_macro)
    metrics["test_loss"] = float(test_loss)
    print("[ViT] Test classification report:")
    print(classification_report(test_labels, test_preds))

    proba_train, y_train = _collect_logits(model, train_eval_loader, device)

    # Persist artifacts
    torch.save(model.state_dict(), out_dir / "model_best.pth")
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(out_dir / "train_proba.npy", proba_train.astype(np.float32))
    np.save(out_dir / "val_proba.npy", proba_val.astype(np.float32))
    np.save(out_dir / "test_proba.npy", proba_test.astype(np.float32))
    np.save(out_dir / "train_labels.npy", y_train.astype(np.int64))
    np.save(out_dir / "val_labels.npy", val_labels)
    np.save(out_dir / "test_labels.npy", test_labels.astype(np.int64))
    np.save(out_dir / "test_preds.npy", test_preds.astype(np.int64))

    train_meta.to_csv(out_dir / "train_metadata.csv", index=False)
    val_meta.to_csv(out_dir / "val_metadata.csv", index=False)
    test_meta.to_csv(out_dir / "test_metadata.csv", index=False)


def main(config_path: str, tune: bool) -> int:
    cfg = _load_config(config_path)
    out_dir = Path(cfg["logging"]["out_dir"]) / "vit"
    run_training(cfg, out_dir, tune=tune)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vit.yaml")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.tune))

