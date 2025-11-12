import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.utils.io import save_json


def _load_modal_results(base: Path, name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    proba_path = base / name / "val_proba.npy"
    labels_path = base / name / "val_labels.npy"
    if not proba_path.exists() or not labels_path.exists():
        return None, None
    proba = np.load(proba_path)
    labels = np.load(labels_path).astype(np.int64)
    return proba, labels


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    pred = proba.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def main(ckpts_dir: str, split: str) -> int:
    base = Path("artifacts")
    results: Dict[str, Dict[str, float]] = {}

    tab_proba, tab_labels = _load_modal_results(base, "tabnet")
    vit_proba, vit_labels = _load_modal_results(base, "vit")
    fusion_proba, fusion_labels = _load_modal_results(base, "fusion")

    reference_labels = None
    if tab_labels is not None:
        reference_labels = tab_labels
    elif vit_labels is not None:
        reference_labels = vit_labels
    elif fusion_labels is not None:
        reference_labels = fusion_labels

    if reference_labels is None:
        print("[WARN] No validation predictions found to evaluate.")
        return 0

    def _check_labels(name: str, labels: np.ndarray | None):
        if labels is None:
            return
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"Label mismatch for {name}; ensure consistent splits across models.")

    if tab_proba is not None:
        _check_labels("tabnet", tab_labels)
        results["tabnet"] = evaluate_predictions(reference_labels, tab_proba)
    if vit_proba is not None:
        _check_labels("vit", vit_labels)
        results["vit"] = evaluate_predictions(reference_labels, vit_proba)
    if fusion_proba is not None:
        _check_labels("fusion", fusion_labels)
        results["fusion"] = evaluate_predictions(reference_labels, fusion_proba)
    if tab_proba is not None and vit_proba is not None:
        avg = (tab_proba + vit_proba) / 2.0
        results["avg_fusion"] = evaluate_predictions(reference_labels, avg)
        results["avg_confusion_matrix"] = confusion_matrix(reference_labels, avg.argmax(axis=1)).tolist()

    out_path = base / "evaluation.json"
    save_json(results, str(out_path))
    print(f"[OK] Wrote evaluation to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=str, default="artifacts/best")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    raise SystemExit(main(args.ckpts, args.split))

