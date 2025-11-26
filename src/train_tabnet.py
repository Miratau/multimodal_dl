import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.ham10000 import build_tabular_preprocessor, make_label_mapping
from src.utils import set_seed
from src.data_augmentation import get_train_val_test_metadata

CONFIG = {
    "seed": 42,
    "device": "cuda",
    "num_workers": 4,
    "data": {
        "raw_dir": "data/ham10000",
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
    "tabnet": {
        "max_epochs": 50,
        "patience": 15,
        "batch_size": 256,
        "virtual_batch_size": 128,
        "params": {
            "n_d": 16,
            "n_a": 16,
            "n_steps": 3,
            "gamma": 1.3,
            "lambda_sparse": 0.0001,
            "lr": 0.02,
            "weight_decay": 0.0001,
        },
    },
}


def _prepare_features(train_df, val_df, target_col):
    label_map = make_label_mapping(train_df, target_col=target_col)
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["label_idx"] = train_df[target_col].map(label_map).astype(int)
    val_df["label_idx"] = val_df[target_col].map(label_map).astype(int)

    exclude_cols = {target_col, "label_idx", "image_path", "image_id"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    preproc = build_tabular_preprocessor(train_df[feature_cols])
    X_train = preproc.fit_transform(train_df[feature_cols])
    X_val = preproc.transform(val_df[feature_cols])

    y_train = train_df["label_idx"].values
    y_val = val_df["label_idx"].values
    return X_train, X_val, y_train, y_val, label_map, preproc


def _train_tabnet(X_train, y_train, X_val, y_val, cfg):
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
    proba_val = clf.predict_proba(X_val).astype(np.float32)
    print(f"[TabNet] Val predictions shape: {proba_val.shape}", flush=True)

    print("[TabNet] Generating predictions on training set...", flush=True)
    proba_train = clf.predict_proba(X_train).astype(np.float32)
    print(f"[TabNet] Train predictions shape: {proba_train.shape}", flush=True)

    print("[TabNet] Computing metrics...", flush=True)
    preds_val = proba_val.argmax(axis=1)
    metrics = {
        "macro_f1": float(f1_score(y_val, preds_val, average="macro")),
    }
    print("[TabNet] Metrics computed.", flush=True)
    return clf, proba_train, proba_val, metrics


def run_training(cfg, out_dir, tune):
    if tune:
        print("[WARN] Tuning disabled; proceeding with fixed TabNet parameters.")
    print("\n[TabNet] Starting training with fixed parameters...", flush=True)
    set_seed(cfg.get("seed", 42))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[TabNet] Loading and splitting metadata...", flush=True)
    val_size = cfg["data"].get("val_size", 0.1)
    test_size = cfg["data"].get("test_size", 0.1)
    train_df, val_df, test_df = get_train_val_test_metadata(
        val_size=val_size,
        test_size=test_size,
        seed=cfg.get("seed", 42),
    )
    print(f"[TabNet] Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}", flush=True)
    target_col = cfg["data"]["target_col"]

    print("[TabNet] Preparing features and preprocessing...", flush=True)
    X_train, X_val, y_train, y_val, label_map, preproc = _prepare_features(train_df, val_df, target_col)
    test_df_enc = test_df.copy()
    test_df_enc["label_idx"] = test_df_enc[target_col].map(label_map).astype(int)
    exclude_cols = {target_col, "label_idx", "image_path", "image_id"}
    feature_cols = [c for c in test_df_enc.columns if c not in exclude_cols]
    X_test = preproc.transform(test_df_enc[feature_cols])
    y_test = test_df_enc["label_idx"].values
    print(f"[TabNet] Feature dimensions: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}", flush=True)

    print("[TabNet] Training model...", flush=True)
    clf, proba_train, proba_val, metrics = _train_tabnet(X_train, y_train, X_val, y_val, cfg)
    preds_val = proba_val.argmax(axis=1)

    print(f"\n[TabNet] Training completed!", flush=True)
    print("[TabNet] Generating predictions on test set...", flush=True)
    proba_test = clf.predict_proba(X_test).astype(np.float32)
    preds_test = proba_test.argmax(axis=1)
    metrics["macro_f1_test"] = float(f1_score(y_test, preds_test, average="macro"))
    print("[TabNet] Test classification report:", flush=True)
    print(classification_report(y_test, preds_test, output_dict=False), flush=True)

    print("[TabNet] Saving artifacts...", flush=True)
    print("[TabNet] Saving model...", flush=True)
    clf.save_model(str(out_dir / "model"))
    print(f"[TabNet] Saved model to {out_dir / 'model'}", flush=True)

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
    np.save(out_dir / "test_proba.npy", proba_test)
    np.save(out_dir / "train_labels.npy", y_train.astype(np.int64))
    np.save(out_dir / "val_labels.npy", y_val.astype(np.int64))
    np.save(out_dir / "test_labels.npy", y_test.astype(np.int64))
    np.save(out_dir / "test_preds.npy", preds_test.astype(np.int64))

    print("[TabNet] Saving metadata CSVs...", flush=True)
    train_df.to_csv(out_dir / "train_metadata.csv", index=False)
    val_df.to_csv(out_dir / "val_metadata.csv", index=False)
    test_df.to_csv(out_dir / "test_metadata.csv", index=False)

    print(f"[TabNet] All artifacts saved to {out_dir}", flush=True)
    print("[TabNet] Done!", flush=True)


def main(tune):
    out_dir = Path(CONFIG["logging"]["out_dir"]) / "tabnet"
    run_training(CONFIG, out_dir, tune=tune)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.tune))
