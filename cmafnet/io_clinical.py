import csv
import os


def _resolve_clinical_path(root):
    data_dir = os.path.join(root, "data")
    for name in os.listdir(data_dir):
        if name.lower().endswith("clinical.csv") or "clinical" in name.lower():
            if name.endswith(".csv"):
                return os.path.join(data_dir, name)
    raise FileNotFoundError("clinical csv")


def _resolve_label_path(root):
    data_dir = os.path.join(root, "data")
    for name in os.listdir(data_dir):
        low = name.lower()
        if low.endswith(".csv") and ("label" in low or "lable" in low or "结局" in name):
            return os.path.join(data_dir, name)
    raise FileNotFoundError("label csv")


def read_clinical_table(root):
    path = _resolve_clinical_path(root)
    for enc in ("gbk", "utf-8-sig", "latin1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(path, b"", 0, 1, "encoding")


def read_label_table(root):
    path = _resolve_label_path(root)
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(path, b"", 0, 1, "encoding")


def clinical_row_vector(row, bleed_key=None):
    from cmafnet.constants import CONTINUOUS_CLINICAL, BINARY_CLINICAL
    out = []
    for k in CONTINUOUS_CLINICAL:
        try:
            out.append(float(row.get(k, 0.0)))
        except (TypeError, ValueError):
            out.append(0.0)
    bleed_col = bleed_key
    if bleed_col is None:
        for k in row.keys():
            if "bleed" in k.lower() or "vaginal" in k.lower():
                bleed_col = k
                break
    for k in BINARY_CLINICAL:
        key = k if k in row else bleed_col
        try:
            out.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            out.append(0.0)
    return out
