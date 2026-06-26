import re


def canonicalize(raw):
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    s = s.replace(".nii.gz", "").replace(".nii", "")
    s = s.replace("_z_mid.png", "").replace(".png", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("(", "").replace(")", "")
    return s


def patient_key(raw):
    c = canonicalize(raw)
    m = re.match(r"^([a-z]*\d+[a-z]?)", c)
    if m:
        return m.group(1)
    m = re.match(r"^(c\d+)", c)
    if m:
        return m.group(1)
    return c.split("-")[0] if "-" in c else c


def orads_index(value):
    if value is None or value == "":
        return -1
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return -1
    mapping = {2: 0, 3: 1, 4: 2, 5: 3}
    return mapping.get(v, -1)


def pathology_index(value):
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return -1
    from cmafnet.constants import PATHOLOGY_TO_IDX
    return PATHOLOGY_TO_IDX.get(v, -1)
