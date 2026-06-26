import argparse

import numpy as np

from cmafnet.metrics.classification import operating_table
from cmafnet.metrics.statistical_tests import delong_auc_test, mcnemar_accuracy
from cmafnet.runtime_paths import project_root, results_path
from cmafnet.train.loop import save_json
import json
import os


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    args = parser.parse_args()
    base = results_path("")
    cmaf_path = os.path.join(base, "internal_test_metrics.json")
    baseline_path = os.path.join(base, "baseline_comparison.json")
    if not os.path.isfile(cmaf_path) or not os.path.isfile(baseline_path):
        return
    cmaf = load_preds(cmaf_path)
    baselines = load_preds(baseline_path)
    out = {}
    for name, metrics in baselines.items():
        out[name] = {
            "mcnemar": {"note": "requires paired prediction store"},
            "macro_auc_delta": cmaf.get("macro_auc", 0) - metrics.get("macro_auc", 0),
        }
    save_json(results_path("statistical_comparisons.json"), out)


if __name__ == "__main__":
    main()
