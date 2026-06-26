import argparse

import numpy as np
import torch

from cmafnet.cohort_manifest import assign_cohort, build_records, load_split_manifest
from cmafnet.constants import SEED_C
from cmafnet.metrics.bootstrap import bootstrap_ci
from cmafnet.metrics.calibration import calibration_stats, decision_curve_net_benefit
from cmafnet.metrics.classification import operating_table
from cmafnet.models.cmafnet import CMAFNet
from cmafnet.normalization import ClinicalNormalizer
from cmafnet.runtime_paths import (
    checkpoint_path,
    external_manifest_path,
    normalizer_path,
    project_root,
    results_path,
    split_manifest_path,
)
from cmafnet.train.checkpoint import load_checkpoint
from cmafnet.train.loop import predict_records, save_json
from cmafnet.transforms import set_global_seed
from cmafnet.viz.panels import dump_metrics_table, plot_calibration, plot_confusion, plot_roc_ovr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_global_seed(SEED_C)
    records = build_records(args.root)
    _, external = assign_cohort(records, external_manifest_path())
    normalizer = ClinicalNormalizer().load(normalizer_path())
    device = torch.device(args.device)
    model = CMAFNet().to(device)
    load_checkpoint(model, checkpoint_path("cmafnet_locked.pt"), device)
    preds = predict_records(model, external, normalizer, device)
    y_true = [p["y_true"] for p in preds]
    y_pred = [p["y_pred"] for p in preds]
    prob = np.array([p["prob"] for p in preds])
    summary = operating_table(y_true, y_pred, prob)
    lo, hi = bootstrap_ci(
        lambda yt, yp, pr: operating_table(yt, yp, pr)["macro_auc"],
        np.array(y_true),
        np.array(y_pred),
        prob,
        seed=SEED_C,
    )
    summary["macro_auc_ci"] = [lo, hi]
    high_risk = prob[:, 1] + prob[:, 2]
    cal = calibration_stats((np.array(y_true) > 0).astype(int), high_risk)
    thresholds = np.linspace(0.05, 0.60, 40)
    nb, ta, tn = decision_curve_net_benefit((np.array(y_true) > 0).astype(int), high_risk, thresholds)
    summary["calibration"] = cal
    summary["decision_curve"] = {"model": nb, "treat_all": ta, "treat_none": tn, "thresholds": thresholds.tolist()}
    dump_metrics_table(summary, results_path("external_metrics.json"))
    plot_confusion(np.array(summary["confusion"]), "External validation", results_path("external_confusion.png"))
    plot_roc_ovr(y_true, prob, results_path("external_roc.png"), "External validation ROC")
    if cal["curve"][0]:
        plot_calibration(cal["curve"][0], cal["curve"][1], results_path("external_calibration.png"))


if __name__ == "__main__":
    main()
