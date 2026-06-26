import argparse

import numpy as np
import torch

from cmafnet.cohort_manifest import build_records, load_split_manifest
from cmafnet.constants import SEED_B
from cmafnet.metrics.classification import operating_table
from cmafnet.models.cmafnet import CMAFNet
from cmafnet.normalization import ClinicalNormalizer
from cmafnet.runtime_paths import checkpoint_path, normalizer_path, project_root, results_path, split_manifest_path
from cmafnet.train.checkpoint import load_checkpoint
from cmafnet.train.loop import predict_records, save_json
from cmafnet.transforms import set_global_seed
from cmafnet.viz.panels import dump_metrics_table, plot_confusion, plot_roc_ovr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_global_seed(SEED_B)
    records = build_records(args.root)
    buckets = load_split_manifest(split_manifest_path(), records)
    normalizer = ClinicalNormalizer().load(normalizer_path())
    device = torch.device(args.device)
    model = CMAFNet().to(device)
    load_checkpoint(model, checkpoint_path("cmafnet_locked.pt"), device)
    preds = predict_records(model, buckets["test"], normalizer, device)
    y_true = [p["y_true"] for p in preds]
    y_pred = [p["y_pred"] for p in preds]
    prob = np.array([p["prob"] for p in preds])
    summary = operating_table(y_true, y_pred, prob)
    lo, hi = __import__("cmafnet.metrics.bootstrap", fromlist=["bootstrap_ci"]).bootstrap_ci(
        lambda yt, yp, pr: operating_table(yt, yp, pr)["macro_auc"],
        np.array(y_true),
        np.array(y_pred),
        prob,
        seed=SEED_B,
    )
    summary["macro_auc_ci"] = [lo, hi]
    dump_metrics_table(summary, results_path("internal_test_metrics.json"))
    plot_confusion(np.array(summary["confusion"]), "Internal test", results_path("internal_confusion.png"))
    plot_roc_ovr(y_true, prob, results_path("internal_roc.png"), "Internal test ROC")


if __name__ == "__main__":
    main()
