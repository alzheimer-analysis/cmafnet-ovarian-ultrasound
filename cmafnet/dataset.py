import torch
from torch.utils.data import Dataset

from cmafnet.transforms import EvalTransform, TrainAugment, load_grayscale


class OvarianMultimodalSet(Dataset):
    def __init__(self, records, normalizer=None, train=False, seed=0):
        self.records = [r for r in records if r.get("image_path")]
        self.normalizer = normalizer
        self.augment = TrainAugment(seed) if train else EvalTransform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        img = load_grayscale(row["image_path"])
        img = self.augment(img)
        x_img = torch.from_numpy(img).unsqueeze(0).float()
        clin = row["clinical"]
        if self.normalizer is not None:
            clin = self.normalizer.transform_row(clin)
        x_clin = torch.tensor(clin, dtype=torch.float32)
        y = torch.tensor(row["pathology"], dtype=torch.long)
        orads = row.get("orads", -1)
        y_orads = torch.tensor(orads, dtype=torch.long)
        return {
            "image": x_img,
            "clinical": x_clin,
            "pathology": y,
            "orads": y_orads,
            "record_id": row["record_id"],
        }
