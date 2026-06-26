import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def one_hot_probs(probs, num_classes=3):
    return probs


def macro_auc(y_true, prob_matrix):
    scores = []
    for c in range(prob_matrix.shape[1]):
        y_bin = (np.array(y_true) == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        scores.append(roc_auc_score(y_bin, prob_matrix[:, c]))
    return float(np.mean(scores)) if scores else 0.0


def per_class_auc(y_true, prob_matrix):
    out = []
    for c in range(prob_matrix.shape[1]):
        y_bin = (np.array(y_true) == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(y_bin, prob_matrix[:, c])))
    return out


def operating_table(y_true, y_pred, prob_matrix):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    rows = []
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        ppv = tp / (tp + fp + 1e-9)
        npv = tn / (tn + fn + 1e-9)
        f1 = f1_score(y_true == c, y_pred == c, zero_division=0)
        auc = per_class_auc(y_true, prob_matrix)[c]
        rows.append(
            {
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "npv": npv,
                "f1": f1,
                "auc": auc,
            }
        )
    macro = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_auc": macro_auc(y_true, prob_matrix),
        "per_class": rows,
        "confusion": cm.tolist(),
    }
    return macro


def summarize_predictions(records):
    y_true = [r["y_true"] for r in records]
    y_pred = [r["y_pred"] for r in records]
    probs = np.array([r["prob"] for r in records])
    return operating_table(y_true, y_pred, probs)
