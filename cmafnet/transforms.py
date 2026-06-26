import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cmafnet.constants import IMG_SIDE


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grayscale(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIDE, IMG_SIDE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.max() > 0:
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr


class TrainAugment:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, arr):
        t = torch.from_numpy(arr).unsqueeze(0)
        if self.rng.random() < 0.5:
            t = TF.hflip(t)
        angle = self.rng.uniform(-12.0, 12.0)
        t = TF.affine(
            t,
            angle=angle,
            translate=(0, 0),
            scale=1.0,
            shear=[self.rng.uniform(-6, 6), self.rng.uniform(-6, 6)],
        )
        if self.rng.random() < 0.35:
            noise = torch.randn_like(t) * 0.04
            t = t + noise
        return t.squeeze(0).numpy()


class EvalTransform:
    def __call__(self, arr):
        return arr
