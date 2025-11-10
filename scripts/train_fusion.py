import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion_mlp import FusionMLP
from src.utils.seed import set_seed
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


def build_dataset(oof_paths, labels: np.ndarray) -> torch.utils.data.Dataset:
    feats = [np.load(p) for p in oof_paths]
    X = np.concatenate(feats, axis=1).astype(np.float32)
    y = labels.astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def cv_score(params: Dict[str, Any], X: np.ndarray, y: np.ndarray, num_classes: int, seed: int = 2025) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = FusionMLP(input_dim=X.shape[1], num_classes=num_classes, hidden_dim=params["hidden_dim"], num_layers=params["num_layers"], dropout=params["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))), batch_size=256, shuffle=True)
        dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.int64))), batch_size=256, shuffle=False)
        best_f1 = -1.0
        epochs_no_improve = 0
        for epoch in range(50):
            model.train()
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            # val
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
            if epochs_no_improve >= 10:
                break
        scores.append(best_f1)
    return float(np.mean(scores))


def main(config_path: str, tune: bool) -> int:
    cfg = _load_config(config_path)
    set_seed(cfg.get("seed", 2025))
    out_dir = Path(cfg["logging"]["out_dir"]) / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load OOF features; expect these files
    tabnet_oof = Path(cfg["logging"]["out_dir"]) / "tabnet" / "oof_proba.npy"
    vit_oof = Path(cfg["logging"]["out_dir"]) / "vit" / "oof_proba.npy"
    if not tabnet_oof.exists() or not vit_oof.exists():
        raise FileNotFoundError("Expected OOF files not found. Run modal trainings first to produce OOF predictions.")
    X_tab = np.load(tabnet_oof)
    X_vit = np.load(vit_oof)
    X = np.concatenate([X_tab, X_vit], axis=1).astype(np.float32)
    # Labels from metadata; assume same ordering during OOF creation
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    y = df[cfg["data"]["target_col"]].astype("category").cat.codes.astype(np.int64).values
    num_classes = int(df[cfg["data"]["target_col"]].nunique())

    if tune:
        def objective(trial: optuna.Trial) -> float:
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
            num_layers = trial.suggest_categorical("num_layers", [1, 2])
            dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
            lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
            params = {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout, "lr": lr}
            score = cv_score(params, X, y, num_classes, seed=cfg.get("seed", 2025))
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=cfg["fusion"]["tune"]["n_trials"], show_progress_bar=True)
        save_json({"best_params": study.best_trial.params, "best_value": study.best_value}, str(out_dir / "study_best.json"))
    else:
        print("Non-tuning training for fusion not yet implemented.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion.yaml")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.tune))

