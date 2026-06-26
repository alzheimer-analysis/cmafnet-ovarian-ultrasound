import torch.nn as nn


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden=128, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)
