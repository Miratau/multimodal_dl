import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score


def evaluate_oof(y_true, proba):
    pred = proba.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def main(ckpts_dir, split):
    base = Path("outputs")
    df = pd.read_csv("data/processed/metadata.csv") if Path("data/processed/metadata.csv").exists() else None
    y_true = None
    if df is not None and "dx" in df.columns:
        y_true = df["dx"].astype("category").cat.codes.values
    results = {}
    
    # TabNet
    tab_oof = base / "tabnet" / "oof_proba.npy"
    if tab_oof.exists() and y_true is not None:
        tab = np.load(tab_oof)
        results["tabnet"] = evaluate_oof(y_true, tab)
    
    # ViT
    vit_oof = base / "vit" / "oof_proba.npy"
    if vit_oof.exists() and y_true is not None:
        vit = np.load(vit_oof)
        results["vit"] = evaluate_oof(y_true, vit)
    
    # Simple average fusion
    if tab_oof.exists() and vit_oof.exists() and y_true is not None:
        avg = (np.load(tab_oof) + np.load(vit_oof)) / 2.0
        results["avg_fusion"] = evaluate_oof(y_true, avg)
        cm = confusion_matrix(y_true, avg.argmax(axis=1)).tolist()
        results["avg_confusion_matrix"] = cm
    
    # Save results
    out_path = base / "evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Wrote evaluation to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=str, default="artifacts/best")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    raise SystemExit(main(args.ckpts, args.split))
