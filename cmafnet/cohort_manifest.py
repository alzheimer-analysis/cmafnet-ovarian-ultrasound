import json
import os
from collections import defaultdict

import numpy as np

from cmafnet.constants import INTERNAL_TEST, INTERNAL_TOTAL, SEED_B
from cmafnet.identifiers import canonicalize, orads_index, pathology_index, patient_key
from cmafnet.io_clinical import clinical_row_vector, read_clinical_table, read_label_table
from cmafnet.io_images import discover_image_index, resolve_image_path


def _label_keys(row):
    keys = list(row.keys())
    return keys[0], keys[1]


def _clinical_bleed_key(row):
    for k in row.keys():
        if "bleed" in k.lower() or "vaginal" in k.lower():
            return k
    return None


def _find_clinical(clinical_by_canon, clinical_by_patient, rid):
    canon = canonicalize(rid)
    if canon in clinical_by_canon:
        return clinical_by_canon[canon]
    for k, v in clinical_by_canon.items():
        if k.startswith(canon) or canon.startswith(k):
            return v
    pid = patient_key(rid)
    rows = clinical_by_patient.get(pid)
    if rows:
        return rows[0]
    return None


def build_records(root):
    labels = read_label_table(root)
    clinical_rows = read_clinical_table(root)
    image_index = discover_image_index(root)

    clinical_by_canon = {}
    clinical_by_patient = defaultdict(list)
    bleed_key = None
    for row in clinical_rows:
        if bleed_key is None:
            bleed_key = _clinical_bleed_key(row)
        c = canonicalize(row["ID"])
        clinical_by_canon[c] = row
        clinical_by_patient[patient_key(row["ID"])].append(row)

    id_key, lab_key = _label_keys(labels[0])
    records = []
    for row in labels:
        rid = row[id_key]
        label = pathology_index(row[lab_key])
        if label < 0:
            continue
        img_path = resolve_image_path(image_index, rid)
        if img_path is None:
            continue
        clin = _find_clinical(clinical_by_canon, clinical_by_patient, rid)
        if clin is None:
            continue
        orads = orads_index(clin.get("orads"))
        orads_raw = clin.get("orads")
        vec = clinical_row_vector(clin, bleed_key=bleed_key)
        records.append(
            {
                "record_id": rid,
                "patient_id": patient_key(rid),
                "image_path": img_path,
                "pathology": label,
                "orads": orads,
                "orads_raw": orads_raw,
                "clinical": vec,
            }
        )
    return records


def assign_cohort(records, external_manifest_path):
    external_ids = set()
    if external_manifest_path and os.path.isfile(external_manifest_path):
        with open(external_manifest_path, encoding="utf-8") as f:
            payload = json.load(f)
        external_ids = set(payload.get("patient_ids", []))
    if not external_ids:
        for r in records:
            if str(r["patient_id"]).startswith("c"):
                external_ids.add(r["patient_id"])
    internal, external = [], []
    for r in records:
        if r["patient_id"] in external_ids:
            external.append(r)
        else:
            internal.append(r)
    return internal, external


def stratified_patient_split(patients, labels, test_count, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for pid, y in zip(patients, labels):
        by_class[y].append(pid)
    test_ids = set()
    for y, pids in by_class.items():
        pids = list(pids)
        rng.shuffle(pids)
        n_test = max(1, round(len(pids) * test_count / len(patients)))
        test_ids.update(pids[:n_test])
    remaining = [p for p in patients if p not in test_ids]
    rng.shuffle(remaining)
    n_val = max(1, int(round(0.15 * len(patients))))
    val_ids = set(remaining[:n_val])
    train_ids = set(remaining[n_val:])
    return train_ids, val_ids, test_ids


def patient_level_split(internal_records, seed=SEED_B, test_n=INTERNAL_TEST):
    patient_label = {}
    for r in internal_records:
        patient_label[r["patient_id"]] = r["pathology"]
    patients = sorted(patient_label.keys())
    labels = [patient_label[p] for p in patients]
    train_ids, val_ids, test_ids = stratified_patient_split(
        patients, labels, test_n, seed
    )
    buckets = {"train": [], "val": [], "test": []}
    for r in internal_records:
        pid = r["patient_id"]
        if pid in test_ids:
            buckets["test"].append(r)
        elif pid in val_ids:
            buckets["val"].append(r)
        else:
            buckets["train"].append(r)
    return buckets


def save_split_manifest(path, buckets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {k: [r["record_id"] for r in v] for k, v in buckets.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_split_manifest(path, records):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    lookup = {r["record_id"]: r for r in records}
    buckets = {}
    for split, ids in payload.items():
        buckets[split] = [lookup[i] for i in ids if i in lookup]
    return buckets
