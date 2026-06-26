import numpy as np
from sklearn.linear_model import LogisticRegression


def calibration_stats(y_true, prob_pos, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    obs = []
    pred = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob_pos >= lo) & (prob_pos < hi if i < n_bins - 1 else prob_pos <= hi)
        if mask.sum() == 0:
            continue
        obs.append(float(np.mean(y_true[mask])))
        pred.append(float(np.mean(prob_pos[mask])))
    if len(obs) < 2:
        return {"intercept": 0.0, "slope": 1.0, "curve": ([], [])}
    x = np.array(pred).reshape(-1, 1)
    y = np.array(obs)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x, y)
    return {
        "intercept": float(lr.intercept_[0]),
        "slope": float(lr.coef_[0][0]),
        "curve": (pred, obs),
    }


def decision_curve_net_benefit(y_true, prob_pos, thresholds):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_pos)
    n = len(y)
    prev = y.mean()
    benefits = []
    for pt in thresholds:
        pred = p >= pt
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        nb = tp / n - fp / n * (pt / (1 - pt + 1e-9))
        benefits.append(float(nb))
    treat_all = [prev - (1 - prev) * (pt / (1 - pt + 1e-9)) for pt in thresholds]
    treat_none = [0.0 for _ in thresholds]
    return benefits, treat_all, treat_none
