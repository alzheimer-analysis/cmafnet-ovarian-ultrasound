import glob
import os

from cmafnet.identifiers import canonicalize, patient_key


def discover_image_index(root):
    data_dir = os.path.join(root, "data")
    folders = ("imagesjpb", "images")
    index = {}
    for folder in folders:
        base = os.path.join(data_dir, folder)
        if not os.path.isdir(base):
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            for path in glob.glob(os.path.join(base, ext)):
                key = canonicalize(os.path.basename(path))
                index[key] = path
    return index


def resolve_image_path(image_index, record_id):
    key = canonicalize(record_id)
    if key in image_index:
        return image_index[key]
    pid = patient_key(record_id)
    candidates = []
    for k, v in image_index.items():
        if k.startswith(key) or key.startswith(k):
            return v
        if patient_key(k) == pid:
            candidates.append(v)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates)[0]
    return None
