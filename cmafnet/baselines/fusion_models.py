import torch
import torch.nn as nn
import timm

from cmafnet.encoders.clinical_mlp import ClinicalEncoder


class ClinicalOnlyHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


class EarlyFusionNet(nn.Module):
    def __init__(self, clinical_dim=9, num_classes=3):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet34", pretrained=False, num_classes=0, in_chans=1
        )
        self.clin = nn.Linear(clinical_dim, 64)
        self.head = nn.Linear(self.encoder.num_features + 64, num_classes)

    def forward(self, image, clinical):
        img = self.encoder(image)
        c = torch.relu(self.clin(clinical))
        z = torch.cat([img, c], dim=-1)
        return self.head(z), None


class LateFusionNet(nn.Module):
    def __init__(self, clinical_dim=9, num_classes=3):
        super().__init__()
        self.img_enc = timm.create_model(
            "resnet34", pretrained=False, num_classes=num_classes, in_chans=1
        )
        self.clin_enc = ClinicalOnlyHead(clinical_dim, num_classes)

    def forward(self, image, clinical):
        li = self.img_enc(image)
        lc = self.clin_enc(clinical)
        return 0.5 * (li + lc), None


class MLPFusionNet(nn.Module):
    def __init__(self, clinical_dim=9, num_classes=3):
        super().__init__()
        self.img_enc = timm.create_model(
            "resnet34", pretrained=False, num_classes=0, in_chans=1
        )
        self.clin_enc = ClinicalEncoder(clinical_dim, 128, 512)
        self.head = nn.Sequential(
            nn.Linear(self.img_enc.num_features + 512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, clinical):
        a = self.img_enc(image)
        b = self.clin_enc(clinical)
        return self.head(torch.cat([a, b], dim=-1)), None


class DAFTModule(nn.Module):
    def __init__(self, channels, tab_dim):
        super().__init__()
        self.scale = nn.Linear(tab_dim, channels)
        self.shift = nn.Linear(tab_dim, channels)

    def forward(self, feat_map, tab):
        g = feat_map
        s = self.scale(tab).unsqueeze(-1).unsqueeze(-1)
        t = self.shift(tab).unsqueeze(-1).unsqueeze(-1)
        return g * (1.0 + s) + t


class DAFTNet(nn.Module):
    def __init__(self, clinical_dim=9, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet34", pretrained=False, features_only=True, in_chans=1
        )
        ch = self.backbone.feature_info.channels()[-1]
        self.daft = DAFTModule(ch, clinical_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, image, clinical):
        feats = self.backbone(image)[-1]
        feats = self.daft(feats, clinical)
        vec = self.pool(feats).flatten(1)
        return self.head(vec), None
