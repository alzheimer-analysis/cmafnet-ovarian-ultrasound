import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image, clinical, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(image, clinical)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return cam.squeeze(1).cpu().numpy()


def extract_embeddings(model, records, normalizer, device):
    from cmafnet.train.loop import build_loader
    loader = build_loader(records, normalizer, train=False)
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            _, _, fused, _ = model(image, clinical, return_features=True)
            feats.append(fused.cpu().numpy())
            labels.extend(batch["pathology"].numpy().tolist())
    return np.concatenate(feats, axis=0), np.array(labels)
