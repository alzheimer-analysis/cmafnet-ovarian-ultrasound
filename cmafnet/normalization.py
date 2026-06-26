import json
import os

import numpy as np

from cmafnet.constants import CONTINUOUS_CLINICAL


class ClinicalNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, records):
        xs = np.array([r["clinical"] for r in records], dtype=np.float64)
        n_cont = len(CONTINUOUS_CLINICAL)
        self.mean = xs[:, :n_cont].mean(axis=0)
        self.std = xs[:, :n_cont].std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform_row(self, vec):
        arr = np.array(vec, dtype=np.float32)
        n_cont = len(CONTINUOUS_CLINICAL)
        arr[:n_cont] = (arr[:n_cont] - self.mean) / self.std
        return arr.tolist()

    def dump(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self.mean = np.array(payload["mean"], dtype=np.float64)
        self.std = np.array(payload["std"], dtype=np.float64)
        return self
