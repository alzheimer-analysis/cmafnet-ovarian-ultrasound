import torch
import torch.nn as nn


class ClinicalOnlyMLP(nn.Module):
    def __init__(self, in_dim=9, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, clinical):
        logits = self.net(clinical)
        return logits, None
