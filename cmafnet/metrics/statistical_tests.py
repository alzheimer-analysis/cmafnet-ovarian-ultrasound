import numpy as np
from scipy import stats


def delong_auc_test(y_true, prob_a, prob_b):
    y = np.asarray(y_true).astype(int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    auc_a = _auc_fast(y, a)
    auc_b = _auc_fast(y, b)
    var = _delong_variance(y, a, b)
    z = (auc_a - auc_b) / (np.sqrt(var) + 1e-9)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "z": float(z), "p": float(p)}


def _auc_fast(y, s):
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos.reshape(-1, 1) - neg.reshape(1, -1)
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _delong_variance(y, s1, s2):
    v = np.var(s1 - s2) / max(len(y), 1)
    return max(v, 1e-8)


def mcnemar_accuracy(y_true, pred_a, pred_b):
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    b01 = np.sum((a == y) & (b != y))
    b10 = np.sum((a != y) & (b == y))
    if b01 + b10 == 0:
        return {"statistic": 0.0, "p": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = 1 - stats.chi2.cdf(stat, 1)
    return {"statistic": float(stat), "p": float(p)}
