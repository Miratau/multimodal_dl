import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.ham10000 import Ham10000Dataset, make_label_mapping, stratified_group_split
from src.models.vit_module import create_vit, split_parameters
from src.training.engine import train_one_epoch, validate
from src.utils.seed import set_seed
from src.utils.transforms import build_train_transforms, build_val_transforms
from src.utils.io import save_json


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


def _sample_params(trial: optuna.Trial, space_cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for key, spec in space_cfg.items():
        if isinstance(spec, list):
            params[key] = trial.suggest_categorical(key, spec)
        elif isinstance(spec, dict) and "loguniform" in spec:
            low, high = spec["loguniform"]
            params[key] = trial.suggest_float(key, low, high, log=True)
        else:
            raise ValueError(f"Unsupported search space for {key}: {spec}")
    # add extra coupled params
    return params


def train_validate_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    backbone: str,
    img_size: int,
    lr_head: float,
    lr_backbone_mult: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
    num_classes: int,
    label_smoothing: float = 0.1,
    amp: bool = True,
) -> np.ndarray:
    train_tf = build_train_transforms(img_size)
    val_tf = build_val_transforms(img_size)
    identity_map = {i: i for i in range(num_classes)}
    ds_tr = Ham10000Dataset(df_tr, class_to_idx=identity_map, image_transform=train_tf, use_tabular=False)
    ds_va = Ham10000Dataset(df_va, class_to_idx=identity_map, image_transform=val_tf, use_tabular=False)
    y_tr = df_tr["dx"].values
    y_va = df_va["dx"].values

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_vit(backbone, num_classes=num_classes, pretrained=True).to(device)
    backbone_params, head_params = split_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_head * lr_backbone_mult},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_f1 = -1.0
    best_proba = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        train_one_epoch(model, dl_tr, criterion, optimizer, device, scaler, use_amp=amp)
        val_loss, proba_va, y_true = validate(model, dl_va, criterion, device)
        pred = proba_va.argmax(axis=1)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_proba = proba_va.copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    return best_proba


def objective_factory(df: pd.DataFrame, cfg: Dict[str, Any], device: torch.device, num_classes: int):
    target_col = cfg["data"]["target_col"]
    group_col = cfg["data"].get("group_col", "lesion_id")
    folds = cfg["data"]["folds"]
    seed = cfg.get("seed", 2025)
    splits = stratified_group_split(df, n_splits=folds, target_col=target_col, group_col=group_col, seed=seed)
    space_cfg = cfg["vit"]["tune"]["params"]
    epochs = cfg["vit"]["epochs"]
    patience = cfg["vit"]["patience"]
    img_size = cfg["vit"]["img_size"]
    label_smoothing = cfg["vit"]["label_smoothing"]
    amp = cfg["vit"]["amp"]

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, space_cfg)
        lr_head = params["lr_head"]
        lr_backbone_mult = params["lr_backbone_mult"]
        weight_decay = params["weight_decay"]
        batch_size = params["batch_size"]
        backbone = params["backbone"]

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        for fold, val_idx in enumerate(splits):
            tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[val_idx].copy()
            proba_va = train_validate_fold(
                df_tr, df_va, backbone, img_size, lr_head, lr_backbone_mult, weight_decay, batch_size, epochs, patience, device, num_classes, label_smoothing, amp
            )
            pred_va = proba_va.argmax(axis=1)
            y_true_all.extend(df_va[target_col].values.tolist())
            y_pred_all.extend(pred_va.tolist())
        macro_f1 = f1_score(np.array(y_true_all), np.array(y_pred_all), average="macro")
        return macro_f1

    return objective


def main(config_path: str, tune: bool) -> int:
    cfg = _load_config(config_path)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 2025))
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    # Encode labels to integers 0..C-1
    label_map = {c: i for i, c in enumerate(sorted(df[cfg["data"]["target_col"]].astype(str).unique()))}
    df[cfg["data"]["target_col"]] = df[cfg["data"]["target_col"]].map(label_map).astype(int)
    num_classes = len(label_map)

    out_dir = Path(cfg["logging"]["out_dir"]) / "vit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if tune:
        study = optuna.create_study(direction="maximize")
        objective = objective_factory(df, cfg, device, num_classes)
        study.optimize(objective, n_trials=cfg["vit"]["tune"]["n_trials"], show_progress_bar=True)
        save_json({"best_params": study.best_trial.params, "best_value": study.best_value}, str(out_dir / "study_best.json"))
    else:
        # Load best params
        best_file = out_dir / "study_best.json"
        if not best_file.exists():
            raise FileNotFoundError(f"{best_file} not found. Run tuning first with --tune.")
        import json
        with open(best_file, "r") as f:
            best = json.load(f)
        params = best["best_params"]
        backbone = params["backbone"]
        lr_head = float(params["lr_head"])
        lr_backbone_mult = float(params["lr_backbone_mult"])
        weight_decay = float(params["weight_decay"])
        batch_size = int(params["batch_size"])
        img_size = int(cfg["vit"]["img_size"])
        epochs = int(cfg["vit"]["epochs"])
        patience = int(cfg["vit"]["patience"])
        label_smoothing = float(cfg["vit"]["label_smoothing"])
        amp = bool(cfg["vit"]["amp"])

        # Build CV and produce OOF probabilities; save fold checkpoints
        splits = stratified_group_split(df, n_splits=cfg["data"]["folds"], target_col=cfg["data"]["target_col"], group_col=cfg["data"].get("group_col", "lesion_id"), seed=cfg.get("seed", 2025))
        oof_proba = np.zeros((len(df), num_classes), dtype=np.float32)
        for fold, val_idx in enumerate(splits):
            tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[val_idx].copy()
            proba_va = train_validate_fold(
                df_tr, df_va, backbone, img_size, lr_head, lr_backbone_mult, weight_decay, batch_size, epochs, patience, device, num_classes, label_smoothing, amp
            )
            oof_proba[val_idx] = proba_va
        np.save(out_dir / "oof_proba.npy", oof_proba)
        print(f"[OK] Saved ViT OOF probabilities to {out_dir/'oof_proba.npy'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vit.yaml")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.tune))

