import argparse

import torch

from cmafnet.cohort_manifest import build_records, load_split_manifest
from cmafnet.constants import SEED_A
from cmafnet.models.cmafnet import CMAFNet
from cmafnet.normalization import ClinicalNormalizer
from cmafnet.runtime_paths import checkpoint_path, normalizer_path, project_root, split_manifest_path
from cmafnet.train.loop import save_json, train_model
from cmafnet.transforms import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_global_seed(SEED_A)
    records = build_records(args.root)
    buckets = load_split_manifest(split_manifest_path(), records)
    normalizer = ClinicalNormalizer().fit(buckets["train"])
    normalizer.dump(normalizer_path())
    device = torch.device(args.device)
    model = CMAFNet().to(device)
    history = train_model(
        model,
        buckets["train"],
        buckets["val"],
        normalizer,
        device,
        checkpoint_path("cmafnet_locked.pt"),
    )
    save_json(checkpoint_path("cmafnet_train_history.json"), history)


if __name__ == "__main__":
    main()
