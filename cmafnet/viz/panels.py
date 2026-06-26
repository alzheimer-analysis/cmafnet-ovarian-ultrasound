import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

CLASS_COLORS = ["#009E73", "#D55E00", "#E69F00"]
CLASS_NAMES = ["Benign", "Malignant", "Borderline"]


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Ben", "Mal", "Bor"])
    ax.set_yticklabels(["Ben", "Mal", "Bor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc_ovr(y_true, prob, path, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for c, ax in enumerate(axes):
        y_bin = (np.array(y_true) == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, prob[:, c])
        ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=1.6)
        ax.plot([0, 1], [0, 1], "--", color="0.7", lw=0.8)
        ax.set_title(CLASS_NAMES[c])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_tsne(features, labels, path, title=""):
    tsne = TSNE(n_components=2, perplexity=30, random_state=17, init="pca")
    emb = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for c in range(3):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.65, color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax.set_title(title)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_calibration(curve_pred, curve_obs, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.plot(curve_pred, curve_obs, "o-", color="#0072B2")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def dump_metrics_table(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
