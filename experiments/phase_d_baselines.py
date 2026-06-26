import argparse
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from cmafnet.baselines.clinical_only import ClinicalOnlyMLP
from cmafnet.baselines.fusion_models import DAFTNet, EarlyFusionNet, LateFusionNet, MLPFusionNet
from cmafnet.baselines.image_only import ImageOnlyClassifier
from cmafnet.clinical_scores.scores import orads_to_pathology_prior, roma_score, roma_to_pathology_prior
from cmafnet.cohort_manifest import build_records, load_split_manifest
from cmafnet.constants import SEED_A
from cmafnet.metrics.classification import operating_table
from cmafnet.models.cmafnet import CMAFNet
from cmafnet.normalization import ClinicalNormalizer
from cmafnet.runtime_paths import checkpoint_path, normalizer_path, project_root, results_path, split_manifest_path
from cmafnet.train.checkpoint import load_checkpoint
from cmafnet.train.loop import predict_records, save_json, train_model
from cmafnet.transforms import set_global_seed


IMAGE_BASELINES = ("resnet50", "efficientnet_b3", "vit_base_patch16_384", "swin_tiny_patch4_window7_224")
FUSION_BASELINES = ("early", "late", "daft", "mlp_fusion")


def clinical_matrix(records):
    return np.array([r["clinical"] for r in records], dtype=np.float32)


def clinical_labels(records):
    return np.array([r["pathology"] for r in records], dtype=int)


def score_orads_roda(records):
    rows = []
    for r in records:
        raw = r.get("orads_raw")
        if raw is None:
            orads = r.get("orads", -1)
            if orads >= 0:
                raw = [2, 3, 4, 5][orads]
            else:
                raw = 3
        prob = np.array(orads_to_pathology_prior(raw), dtype=float)
        pred = int(prob.argmax())
        rows.append({"y_true": r["pathology"], "y_pred": pred, "prob": prob})
    return rows


def score_roma(records):
    rows = []
    for r in records:
        vec = r["clinical"]
        he4, ca125, post = vec[5], vec[2], vec[7]
        rs = roma_score(he4, ca125, post)
        prob = np.array(roma_to_pathology_prior(rs, post), dtype=float)
        pred = int(prob.argmax())
        rows.append({"y_true": r["pathology"], "y_pred": pred, "prob": prob})
    return rows


def train_sklearn_clinical(train, test):
    x_train = clinical_matrix(train)
    y_train = clinical_labels(train)
    x_test = clinical_matrix(test)
    y_test = clinical_labels(test)
    out = {}
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
    lr.fit(x_train, y_train)
    prob = lr.predict_proba(x_test)
    pred = prob.argmax(axis=1)
    out["logistic_regression"] = operating_table(y_test, pred, prob)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=SEED_A,
    )
    xgb.fit(x_train, y_train)
    prob = xgb.predict_proba(x_test)
    pred = prob.argmax(axis=1)
    out["xgboost"] = operating_table(y_test, pred, prob)
    with open(results_path("sklearn_clinical.pkl"), "wb") as f:
        pickle.dump({"lr": lr, "xgb": xgb}, f)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-heavy", action="store_true")
    args = parser.parse_args()
    set_global_seed(SEED_A)
    records = build_records(args.root)
    buckets = load_split_manifest(split_manifest_path(), records)
    normalizer = ClinicalNormalizer().load(normalizer_path())
    device = torch.device(args.device)
    test = buckets["test"]
    results = {}
    results["orads"] = operating_table(
        [r["y_true"] for r in score_orads_roda(test)],
        [r["y_pred"] for r in score_orads_roda(test)],
        np.array([r["prob"] for r in score_orads_roda(test)]),
    )
    results["roma"] = operating_table(
        [r["y_true"] for r in score_roma(test)],
        [r["y_pred"] for r in score_roma(test)],
        np.array([r["prob"] for r in score_roma(test)]),
    )
    results.update(train_sklearn_clinical(buckets["train"], test))
    if not args.skip_heavy:
        for backbone in IMAGE_BASELINES:
            model = ImageOnlyClassifier(backbone).to(device)
            ck = checkpoint_path(f"image_{backbone}.pt")
            train_model(model, buckets["train"], buckets["val"], normalizer, device, ck, epochs=40, seed=SEED_A)
            load_checkpoint(model, ck, device)
            preds = predict_records(model, test, normalizer, device)
            prob = np.array([p["prob"] for p in preds])
            results[backbone] = operating_table(
                [p["y_true"] for p in preds], [p["y_pred"] for p in preds], prob
            )
        clin_mlp = ClinicalOnlyMLP().to(device)
        ck = checkpoint_path("clinical_mlp.pt")
        train_model(clin_mlp, buckets["train"], buckets["val"], normalizer, device, ck, epochs=60, seed=SEED_A)
        load_checkpoint(clin_mlp, ck, device)
        preds = predict_records(clin_mlp, test, normalizer, device)
        prob = np.array([p["prob"] for p in preds])
        results["clinical_mlp"] = operating_table(
            [p["y_true"] for p in preds], [p["y_pred"] for p in preds], prob
        )
        fusion_map = {
            "early": EarlyFusionNet,
            "late": LateFusionNet,
            "daft": DAFTNet,
            "mlp_fusion": MLPFusionNet,
        }
        for name, cls in fusion_map.items():
            model = cls().to(device)
            ck = checkpoint_path(f"fusion_{name}.pt")
            train_model(model, buckets["train"], buckets["val"], normalizer, device, ck, epochs=50, seed=SEED_A)
            load_checkpoint(model, ck, device)
            preds = predict_records(model, test, normalizer, device)
            prob = np.array([p["prob"] for p in preds])
            results[name] = operating_table(
                [p["y_true"] for p in preds], [p["y_pred"] for p in preds], prob
            )
    save_json(results_path("baseline_comparison.json"), results)


if __name__ == "__main__":
    main()
