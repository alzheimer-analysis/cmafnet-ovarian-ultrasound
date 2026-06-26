import torch
import torch.nn as nn
import torch.nn.functional as F

from cmafnet.constants import AUX_WEIGHT, FOCAL_GAMMA, NUM_ORADS, NUM_PATHOLOGY


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=FOCAL_GAMMA):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("alpha", class_weights.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets_oh = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_oh).sum(dim=-1)
        log_pt = (log_probs * targets_oh).sum(dim=-1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            aw = self.alpha[targets]
            loss = loss * aw
        return loss.mean()


class MultitaskLoss(nn.Module):
    def __init__(self, class_weights=None, aux_weight=AUX_WEIGHT):
        super().__init__()
        self.primary = WeightedFocalLoss(class_weights)
        self.aux = nn.CrossEntropyLoss(ignore_index=-1)
        self.aux_weight = aux_weight

    def forward(self, logits, orads_logits, targets, orads_targets):
        loss = self.primary(logits, targets)
        if orads_logits is not None:
            mask = orads_targets >= 0
            if mask.any():
                aux = self.aux(orads_logits[mask], orads_targets[mask])
                loss = loss + self.aux_weight * aux
        return loss
