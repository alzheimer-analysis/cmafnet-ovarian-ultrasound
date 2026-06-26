import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmafnet.constants import BATCH_SIZE, LR, NUM_WORKERS, TRAIN_EPOCHS, WEIGHT_DECAY
from cmafnet.dataset import OvarianMultimodalSet
from cmafnet.losses.focal import MultitaskLoss


def class_weights_from_records(records):
    counts = np.bincount([r["pathology"] for r in records], minlength=3).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv / inv.sum() * 3.0
    return torch.tensor(w, dtype=torch.float32)


def build_loader(records, normalizer, train=False, batch_size=BATCH_SIZE, seed=0):
    ds = OvarianMultimodalSet(records, normalizer=normalizer, train=train, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def train_model(
    model,
    train_records,
    val_records,
    normalizer,
    device,
    save_path,
    epochs=TRAIN_EPOCHS,
    seed=0,
    use_focal=True,
):
    train_loader = build_loader(train_records, normalizer, train=True, seed=seed)
    val_loader = build_loader(val_records, normalizer, train=False, seed=seed)
    weights = class_weights_from_records(train_records).to(device)
    criterion = MultitaskLoss(class_weights=weights if use_focal else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            y = batch["pathology"].to(device)
            orads = batch["orads"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, orads_logits = model(image, clinical)
            loss = criterion(logits, orads_logits, y, orads)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        val_metrics = evaluate_loader(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": val_metrics,
                },
                save_path,
            )
    return history


@torch.no_grad()
def evaluate_loader(model, loader, device):
    from cmafnet.metrics.classification import operating_table
    model.eval()
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        clinical = batch["clinical"].to(device)
        logits, _ = model(image, clinical)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = prob.argmax(axis=-1)
        y = batch["pathology"].numpy()
        for i in range(len(y)):
            rows.append({"y_true": int(y[i]), "y_pred": int(pred[i]), "prob": prob[i]})
    summary = operating_table(
        [r["y_true"] for r in rows],
        [r["y_pred"] for r in rows],
        np.array([r["prob"] for r in rows]),
    )
    return summary


@torch.no_grad()
def predict_records(model, records, normalizer, device, batch_size=BATCH_SIZE):
    loader = build_loader(records, normalizer, train=False, batch_size=batch_size)
    model.eval()
    outputs = []
    for batch in loader:
        image = batch["image"].to(device)
        clinical = batch["clinical"].to(device)
        logits, _ = model(image, clinical)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = prob.argmax(axis=-1)
        y = batch["pathology"].numpy()
        ids = batch["record_id"]
        for i in range(len(y)):
            outputs.append(
                {
                    "record_id": ids[i],
                    "y_true": int(y[i]),
                    "y_pred": int(pred[i]),
                    "prob": prob[i],
                }
            )
    return outputs


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
