import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import torch

from src.data.ham10000 import (
    build_tabular_preprocessor,
    make_label_mapping,
    stratified_group_split,
)
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import compute_classification_metrics
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.utils.class_weight import compute_class_weight


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    include = cfg.get("include")
    if include:
        with open(include, "r") as f:
            base = yaml.safe_load(f)
        # naive merge: base <- cfg (override)
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
    return params


def _fit_one_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    target_col: str,
    params: Dict[str, Any],
    max_epochs: int,
    patience: int,
) -> Tuple[np.ndarray, np.ndarray]:
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

    # Metrics: avoid AUC for multiclass to prevent sklearn error
    eval_metrics = ["balanced_accuracy", "logloss"]

    clf = TabNetClassifier(
        n_d=params.get("n_d", 32),
        n_a=params.get("n_a", 32),
        n_steps=params.get("n_steps", 5),
        gamma=params.get("gamma", 1.5),
        lambda_sparse=params.get("lambda_sparse", 0.0),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 1e-5)),
        mask_type="sparsemax",
        verbose=0,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric=eval_metrics,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=params.get("batch_size", 512),
        virtual_batch_size=params.get("virtual_batch_size", 128),
        weights=sample_weights,
    )
    proba_va = clf.predict_proba(X_va).astype(np.float32)
    pred_va = proba_va.argmax(axis=1)
    return pred_va, proba_va


def objective_factory(df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path):
    target_col = cfg["data"]["target_col"]
    group_col = cfg["data"].get("group_col", "lesion_id")
    folds = cfg["data"]["folds"]
    seed = cfg.get("seed", 2025)
    splits = stratified_group_split(df, n_splits=folds, target_col=target_col, group_col=group_col, seed=seed)

    def objective(trial: optuna.Trial) -> float:
        space_cfg = cfg["tabnet"]["tune"]["params"]
        params = _sample_params(trial, space_cfg)
        params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
        max_epochs = cfg["tabnet"]["max_epochs"]
        patience = cfg["tabnet"]["patience"]

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        for fold, val_idx in enumerate(splits):
            tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[val_idx].copy()
            pred_va, proba_va = _fit_one_fold(df_tr, df_va, target_col, params, max_epochs, patience)
            y_true_all.extend(df_va[target_col].values.tolist())
            y_pred_all.extend(pred_va.tolist())
        macro_f1 = f1_score(np.array(y_true_all), np.array(y_pred_all), average="macro")
        return macro_f1

    return objective


def run_training(df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path, tune: bool) -> None:
    set_seed(cfg.get("seed", 2025))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Encode labels to integers
    label_map = make_label_mapping(df, target_col=cfg["data"]["target_col"])
    df_ = df.copy()
    df_[cfg["data"]["target_col"]] = df_[cfg["data"]["target_col"]].map(label_map).astype(int)

    if tune:
        study = optuna.create_study(direction="maximize")
        objective = objective_factory(df_, cfg, out_dir)
        study.optimize(objective, n_trials=cfg["tabnet"]["tune"]["n_trials"], show_progress_bar=True)
        best_params = study.best_trial.params
        save_json({"best_params": best_params, "best_value": study.best_value}, str(out_dir / "study_best.json"))
    else:
        # Train with default params and save simple OOF predictions
        splits = stratified_group_split(
            df_, n_splits=cfg["data"]["folds"], target_col=cfg["data"]["target_col"], group_col=cfg["data"].get("group_col", "lesion_id"), seed=cfg.get("seed", 2025)
        )
        oof_proba = np.zeros((len(df_), len(label_map)), dtype=np.float32)
        for fold, val_idx in enumerate(splits):
            tr_idx = np.setdiff1d(np.arange(len(df_)), val_idx)
            df_tr = df_.iloc[tr_idx].copy()
            df_va = df_.iloc[val_idx].copy()
            params = {
                "n_d": 32, "n_a": 32, "n_steps": 5, "gamma": 1.5, "lambda_sparse": 1e-5, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": cfg["tabnet"]["batch_size"], "virtual_batch_size": cfg["tabnet"]["virtual_batch_size"]
            }
            pred_va, proba_va = _fit_one_fold(df_tr, df_va, cfg["data"]["target_col"], params, cfg["tabnet"]["max_epochs"], cfg["tabnet"]["patience"])
            oof_proba[val_idx] = proba_va
        np.save(out_dir / "oof_proba.npy", oof_proba)
        print(f"[OK] Saved OOF probabilities to {out_dir/'oof_proba.npy'}")


def main(config_path: str, tune: bool) -> int:
    cfg = _load_config(config_path)
    data_csv = cfg["data"]["metadata_csv"]
    out_dir = Path(cfg["logging"]["out_dir"]) / "tabnet"
    df = pd.read_csv(data_csv)
    run_training(df, cfg, out_dir, tune=tune)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tabnet.yaml")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.tune))

