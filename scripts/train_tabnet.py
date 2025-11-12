import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch

from src.data.ham10000 import build_tabular_preprocessor, make_label_mapping
from src.utils.seed import set_seed

from multimodal_dl.utils import get_train_val_metadata


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


def _prepare_features(
    train_df,
    val_df,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Any]:
    label_map = make_label_mapping(train_df, target_col=target_col)
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["label_idx"] = train_df[target_col].map(label_map).astype(int)
    val_df["label_idx"] = val_df[target_col].map(label_map).astype(int)

    exclude_cols = {target_col, "label_idx", "image_path", "image_id"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    preproc = build_tabular_preprocessor(train_df[feature_cols])
    X_train = preproc.transform(train_df[feature_cols])
    X_val = preproc.transform(val_df[feature_cols])

    y_train = train_df["label_idx"].values
    y_val = val_df["label_idx"].values
    return X_train, X_val, y_train, y_val, label_map, preproc


def _train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> tuple[TabNetClassifier, np.ndarray, np.ndarray, Dict[str, float]]:
    params_cfg = cfg["tabnet"]
    fixed_params = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 5,
        "gamma": 1.5,
        "lambda_sparse": 1e-5,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": params_cfg.get("batch_size", 512),
        "virtual_batch_size": params_cfg.get("virtual_batch_size", 128),
    }
    fixed_params.update(params_cfg.get("params", {}))

    classes = np.unique(y_train)
    class_weights_vec = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_to_w = {c: w for c, w in zip(classes, class_weights_vec)}
    sample_weights = np.array([class_to_w[c] for c in y_train], dtype=np.float32)

    clf = TabNetClassifier(
        n_d=int(fixed_params["n_d"]),
        n_a=int(fixed_params["n_a"]),
        n_steps=int(fixed_params["n_steps"]),
        gamma=float(fixed_params["gamma"]),
        lambda_sparse=float(fixed_params["lambda_sparse"]),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(
            lr=float(fixed_params.get("lr", 1e-3)),
            weight_decay=float(fixed_params.get("weight_decay", 1e-5)),
        ),
        mask_type="sparsemax",
        verbose=1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["balanced_accuracy", "logloss"],
        max_epochs=params_cfg.get("max_epochs", 50),
        patience=params_cfg.get("patience", 10),
        batch_size=int(fixed_params["batch_size"]),
        virtual_batch_size=int(fixed_params["virtual_batch_size"]),
        weights=sample_weights,
    )

    print("[TabNet] Generating predictions on validation set...", flush=True)
    try:
        proba_val = clf.predict_proba(X_val).astype(np.float32)
        print(f"[TabNet] Val predictions shape: {proba_val.shape}", flush=True)
    except Exception as e:
        print(f"[TabNet] ERROR generating val predictions: {e}", flush=True)
        raise
    
    print("[TabNet] Generating predictions on training set...", flush=True)
    try:
        proba_train = clf.predict_proba(X_train).astype(np.float32)
        print(f"[TabNet] Train predictions shape: {proba_train.shape}", flush=True)
    except Exception as e:
        print(f"[TabNet] ERROR generating train predictions: {e}", flush=True)
        raise
    
    print("[TabNet] Computing metrics...", flush=True)
    preds_val = proba_val.argmax(axis=1)
    metrics = {
        "macro_f1": float(f1_score(y_val, preds_val, average="macro")),
    }
    print("[TabNet] Metrics computed.", flush=True)
    return clf, proba_train, proba_val, metrics


def run_training(cfg: Dict[str, Any], out_dir: Path, tune: bool) -> None:
    try:
        if tune:
            print("[WARN] Tuning disabled; proceeding with fixed TabNet parameters.")
        print("\n[TabNet] Starting training with fixed parameters...", flush=True)
        set_seed(cfg.get("seed", 2025))
        out_dir.mkdir(parents=True, exist_ok=True)

        print("[TabNet] Loading and splitting metadata...", flush=True)
        test_size = cfg["data"].get("test_size", 0.2)
        train_df, val_df = get_train_val_metadata(
            test_size=test_size,
            seed=cfg.get("seed", 2025),
        )
        print(f"[TabNet] Train samples: {len(train_df)}, Val samples: {len(val_df)}", flush=True)
        target_col = cfg["data"]["target_col"]

        print("[TabNet] Preparing features and preprocessing...", flush=True)
        X_train, X_val, y_train, y_val, label_map, preproc = _prepare_features(train_df, val_df, target_col)
        print(f"[TabNet] Feature dimensions: train={X_train.shape}, val={X_val.shape}", flush=True)
        
        print("[TabNet] Training model...", flush=True)
        clf, proba_train, proba_val, metrics = _train_tabnet(X_train, y_train, X_val, y_val, cfg)
        preds_val = proba_val.argmax(axis=1)

        print(f"\n[TabNet] Training completed!", flush=True)
        print(f"[TabNet] Macro F1 on validation: {metrics['macro_f1']:.4f}", flush=True)
        print("[TabNet] Classification report:", flush=True)
        report = classification_report(y_val, preds_val, output_dict=False)
        print(report, flush=True)

        # Persist artifacts
        print("[TabNet] Saving artifacts...", flush=True)
        try:
            print("[TabNet] Saving model...", flush=True)
            clf.save_model(str(out_dir / "model"))
            print(f"[TabNet] Saved model to {out_dir / 'model'}", flush=True)
        except Exception as e:
            print(f"[TabNet] ERROR saving model: {e}", flush=True)
            raise
        
        print("[TabNet] Saving preprocessor...", flush=True)
        with open(out_dir / "tabular_preprocessor.pkl", "wb") as f:
            pickle.dump(preproc, f)
        
        print("[TabNet] Saving label map and metrics...", flush=True)
        with open(out_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("[TabNet] Saving probabilities and labels...", flush=True)
        np.save(out_dir / "train_proba.npy", proba_train)
        np.save(out_dir / "val_proba.npy", proba_val)
        np.save(out_dir / "train_labels.npy", y_train.astype(np.int64))
        np.save(out_dir / "val_labels.npy", y_val.astype(np.int64))

        print("[TabNet] Saving metadata CSVs...", flush=True)
        train_df.to_csv(out_dir / "train_metadata.csv", index=False)
        val_df.to_csv(out_dir / "val_metadata.csv", index=False)
        
        print(f"[TabNet] All artifacts saved to {out_dir}", flush=True)
        print("[TabNet] Done!", flush=True)
    except Exception as e:
        print(f"\n[TabNet] FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def main(config_path: str, tune: bool) -> int:
    cfg = _load_config(config_path)
    out_dir = Path(cfg["logging"]["out_dir"]) / "tabnet"
    run_training(cfg, out_dir, tune=tune)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tabnet.yaml")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.tune))

