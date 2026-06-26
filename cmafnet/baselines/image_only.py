import torch
import torch.nn as nn
import timm


def _adapt_first_conv(model, in_ch=1):
    conv = model.conv1
    if conv.in_channels == in_ch:
        return model
    weight = conv.weight.data.mean(dim=1, keepdim=True)
    new_conv = nn.Conv2d(
        in_ch,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )
    new_conv.weight.data = weight
    model.conv1 = new_conv
    return model


class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone, num_classes=3, pretrained=False):
        super().__init__()
        if backbone == "resnet50":
            self.encoder = timm.create_model(
                "resnet50", pretrained=pretrained, num_classes=0, in_chans=1
            )
            dim = self.encoder.num_features
        elif backbone == "efficientnet_b3":
            self.encoder = timm.create_model(
                "efficientnet_b3", pretrained=pretrained, num_classes=0, in_chans=1
            )
            dim = self.encoder.num_features
        elif backbone == "vit_base_patch16_384":
            self.encoder = timm.create_model(
                "vit_base_patch16_384", pretrained=pretrained, num_classes=0, in_chans=1
            )
            dim = self.encoder.num_features
        elif backbone == "swin_tiny_patch4_window7_224":
            self.encoder = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0,
                in_chans=1,
                img_size=384,
            )
            dim = self.encoder.num_features
        else:
            raise ValueError(backbone)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, image, clinical=None):
        feat = self.encoder(image)
        logits = self.head(feat)
        return logits, None
