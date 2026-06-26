import argparse

import numpy as np
import torch

from cmafnet.cohort_manifest import build_records, load_split_manifest
from cmafnet.constants import SEED_A
from cmafnet.metrics.classification import operating_table
from cmafnet.models.cmafnet import CMAFNet, build_ablation_variant
from cmafnet.normalization import ClinicalNormalizer
from cmafnet.runtime_paths import checkpoint_path, normalizer_path, project_root, results_path, split_manifest_path
from cmafnet.train.checkpoint import load_checkpoint
from cmafnet.train.loop import predict_records, save_json, train_model
from cmafnet.transforms import set_global_seed


ABLATIONS = (
    "full",
    "no_cross",
    "no_gate",
    "cnn_only",
    "transformer_only",
    "no_orads",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variant", default="all")
    args = parser.parse_args()
    set_global_seed(SEED_A)
    records = build_records(args.root)
    buckets = load_split_manifest(split_manifest_path(), records)
    normalizer = ClinicalNormalizer().load(normalizer_path())
    device = torch.device(args.device)
    test = buckets["test"]
    variants = ABLATIONS if args.variant == "all" else (args.variant,)
    out = {}
    for name in variants:
        if name == "full":
            model = CMAFNet().to(device)
        else:
            model = build_ablation_variant(name).to(device)
        ck = checkpoint_path(f"ablation_{name}.pt")
        use_focal = name != "no_orads"
        train_model(
            model,
            buckets["train"],
            buckets["val"],
            normalizer,
            device,
            ck,
            epochs=60 if name != "full" else 80,
            seed=SEED_A,
            use_focal=use_focal,
        )
        load_checkpoint(model, ck, device)
        preds = predict_records(model, test, normalizer, device)
        prob = np.array([p["prob"] for p in preds])
        out[name] = operating_table(
            [p["y_true"] for p in preds], [p["y_pred"] for p in preds], prob
        )
    save_json(results_path("ablation_results.json"), out)


if __name__ == "__main__":
    main()
