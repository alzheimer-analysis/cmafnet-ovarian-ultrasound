import numpy as np


def bootstrap_ci(values_fn, y_true, y_pred, prob, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(values_fn(y_true[idx], y_pred[idx], prob[idx]))
    arr = np.array(stats, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_metric_fn(metric_fn):
    def wrapped(y_true, y_pred, prob):
        return metric_fn(y_true, y_pred, prob)
    return wrapped
