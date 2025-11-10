from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    metrics = {}
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    try:
        num_classes = y_proba.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]
        metrics["macro_auroc"] = roc_auc_score(y_true_onehot, y_proba, average="macro", multi_class="ovr")
    except Exception:
        metrics["macro_auroc"] = float("nan")
    return metrics

