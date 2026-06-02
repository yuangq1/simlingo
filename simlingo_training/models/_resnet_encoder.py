"""
ResNet-34 encoder for TCP models.
Modified forward returns both FC embedding and spatial feature map.
Adapted from original TCP resnet.py — works with any input resolution.
"""

import torch
from torch import nn
from torchvision.models import resnet34
from typing import Tuple


class ResNetEncoder(nn.Module):
    """ResNet-34 that returns (feature_emb [B,1000], cnn_feature [B,512,H',W'])."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet34(weights='DEFAULT' if pretrained else None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc  # 512 → 1000

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] RGB image, normalized

        Returns:
            feature_emb: [B, 1000]  FC-level embedding
            cnn_feature: [B, 512, h, w]  layer4 spatial feature map
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_layer4 = self.layer4(x)

        x_pooled = self.avgpool(x_layer4)
        x_flat = torch.flatten(x_pooled, 1)
        feature_emb = self.fc(x_flat)

        return feature_emb, x_layer4
