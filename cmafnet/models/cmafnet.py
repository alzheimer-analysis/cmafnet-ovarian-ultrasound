import torch
import torch.nn as nn
import torch.nn.functional as F

from cmafnet.constants import (
    AUX_WEIGHT,
    CLIN_HIDDEN,
    EMBED_DIM,
    FOCAL_GAMMA,
    FUSION_DIM,
    HEAD_DROPOUT,
    NUM_ORADS,
    NUM_PATHOLOGY,
)
from cmafnet.encoders.clinical_mlp import ClinicalEncoder
from cmafnet.encoders.hybrid_image import HybridImageEncoder
from cmafnet.fusion.cross_attention import BidirectionalCrossAttention, GatedFusion


class CMAFNet(nn.Module):
    def __init__(
        self,
        clinical_dim=9,
        embed_dim=EMBED_DIM,
        use_cross_attention=True,
        use_gated_fusion=True,
        use_hybrid_cnn=True,
        use_hybrid_transformer=True,
        use_orads_head=True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_gated_fusion = use_gated_fusion
        self.use_hybrid_cnn = use_hybrid_cnn
        self.use_hybrid_transformer = use_hybrid_transformer
        self.use_orads_head = use_orads_head
        self.image_encoder = HybridImageEncoder(out_dim=embed_dim)
        self.clinical_encoder = ClinicalEncoder(
            in_dim=clinical_dim, hidden=CLIN_HIDDEN, out_dim=embed_dim
        )
        self.cross = BidirectionalCrossAttention(embed_dim)
        self.gated = GatedFusion(FUSION_DIM)
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.pathology_head = nn.Linear(FUSION_DIM, NUM_PATHOLOGY)
        self.orads_head = nn.Linear(FUSION_DIM, NUM_ORADS)

    def encode_image(self, x):
        feat_map, vec = self.image_encoder(x)
        if not self.use_hybrid_transformer:
            vec = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        if not self.use_hybrid_cnn:
            tokens = self.image_encoder.patch(self.image_encoder.proj(feat_map))
            b, c, h, w = tokens.shape
            seq = tokens.flatten(2).transpose(1, 2)
            for blk in self.image_encoder.blocks:
                seq = blk(seq)
            vec = seq.mean(dim=1)
        return feat_map, vec

    def forward(self, image, clinical, return_features=False):
        feat_map, img_vec = self.encode_image(image)
        clin_vec = self.clinical_encoder(clinical)
        if self.use_cross_attention:
            feat_map, clin_vec = self.cross(feat_map, clin_vec)
            img_vec = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        if self.use_gated_fusion:
            fused = self.gated(img_vec, clin_vec)
        else:
            fused = 0.5 * (img_vec + clin_vec)
        fused = self.dropout(fused)
        logits = self.pathology_head(fused)
        orads_logits = self.orads_head(fused) if self.use_orads_head else None
        if return_features:
            return logits, orads_logits, fused, feat_map
        return logits, orads_logits


def build_ablation_variant(name):
    variants = {
        "full": dict(),
        "no_cross": dict(use_cross_attention=False),
        "no_clin_to_img": dict(use_cross_attention=True),
        "no_img_to_clin": dict(use_cross_attention=True),
        "no_gate": dict(use_gated_fusion=False),
        "cnn_only": dict(use_hybrid_transformer=False),
        "transformer_only": dict(use_hybrid_cnn=False),
        "no_orads": dict(use_orads_head=False),
        "no_clinical": None,
    }
    cfg = variants.get(name, dict())
    if cfg is None:
        return None
    return CMAFNet(**cfg)
