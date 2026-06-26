import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from cmafnet.cohort_manifest import build_records, load_split_manifest
from cmafnet.constants import SEED_A
from cmafnet.models.cmafnet import CMAFNet
from cmafnet.normalization import ClinicalNormalizer
from cmafnet.runtime_paths import checkpoint_path, normalizer_path, project_root, results_path, split_manifest_path
from cmafnet.train.checkpoint import load_checkpoint
from cmafnet.transforms import load_grayscale, set_global_seed
from cmafnet.viz.interpret import GradCAM, extract_embeddings
from cmafnet.viz.panels import plot_tsne


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--case-limit", type=int, default=6)
    args = parser.parse_args()
    set_global_seed(SEED_A)
    records = build_records(args.root)
    buckets = load_split_manifest(split_manifest_path(), records)
    normalizer = ClinicalNormalizer().load(normalizer_path())
    device = torch.device(args.device)
    model = CMAFNet().to(device)
    load_checkpoint(model, checkpoint_path("cmafnet_locked.pt"), device)
    model.eval()
    target_layer = model.image_encoder.layer4[-1]
    cam = GradCAM(model, target_layer)
    out_dir = os.path.join(results_path("interpret"), "gradcam")
    os.makedirs(out_dir, exist_ok=True)
    subset = buckets["test"][: args.case_limit]
    for row in subset:
        img = load_grayscale(row["image_path"])
        clin = normalizer.transform_row(row["clinical"])
        x_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
        x_clin = torch.tensor(clin, dtype=torch.float32).unsqueeze(0).to(device)
        pred_cls = int(torch.softmax(model(x_img, x_clin)[0], dim=-1).argmax().item())
        heat = cam(x_img, x_clin, pred_cls)[0]
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(row["record_id"])
        axes[0].axis("off")
        axes[1].imshow(img, cmap="gray")
        axes[1].imshow(heat, cmap="jet", alpha=0.45)
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{row['record_id']}.png"), dpi=200)
        plt.close(fig)
    feats, labels = extract_embeddings(model, buckets["test"], normalizer, device)
    plot_tsne(feats, labels, results_path("interpret/tsne_cmafnet.png"), "CMAFNet fused features")


if __name__ == "__main__":
    main()
