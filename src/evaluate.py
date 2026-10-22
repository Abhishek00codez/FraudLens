import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score
)

def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.05) -> Tuple[float, Dict[str, float]]:
    # sweep decision thresholds
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    best_f2 = -1.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        # calculate TN, FP, FN, TP
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()

        # calculate False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        if fpr <= max_fpr:
            # optimize for F2 score (recall weighted more heavily than precision)
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

            if f2 > best_f2:
                best_f2 = f2
                best_thresh = thresh

    # final evaluation at best threshold
    y_pred_best = (y_prob >= best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm_best.ravel()

    metrics = {
        'threshold': best_thresh,
        'f2_score': best_f2,
        'f1_score': f1_score(y_true, y_pred_best, zero_division=0),
        'precision': precision_score(y_true, y_pred_best, zero_division=0),
        'recall': recall_score(y_true, y_pred_best, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

    return best_thresh, metrics

def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None) -> Dict[str, float]:
    if threshold is None:
        threshold, metrics = optimize_threshold(y_true, y_prob)
        return metrics

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'threshold': threshold,
        'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

    return metrics

if __name__ == "__main__":
    # Test with dummy data
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.4, 0.2, 0.5, 0.3, 0.2, 0.1, 0.8, 0.9])

    thresh, metrics = optimize_threshold(y_true, y_prob, max_fpr=0.1)
    print(f"Optimal Threshold: {thresh:.4f}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
