"""
The network behind the published numbers, lifted out of the notebook.

It is not a plain ResNet-50. Two tracks look at the image side by side:

  1. A ResNet-50 trunk pretrained on ImageNet, with its pooling and classifier
     removed, giving a 2048-channel feature map.
  2. A small CNN built from an Inception block, two Ghost blocks with ECA
     channel attention, and a 1x1 expansion to 1024 channels.

The ResNet map (7x7 for a 224 input) is resampled to the second track's 56x56,
the two are concatenated (3072 channels), re-weighted by a squeeze-and-excitation gate, then
globally pooled and classified by a three-layer head with dropout.

Every module is as the notebook defined it. The only change is that the
pretrained trunk is loaded through the current torchvision weights API, and
`pretrained=False` is available for tests that must not download anything.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 37


class GhostModule(nn.Module):
    """Half the output channels from a real convolution, the other half from a cheap depthwise one."""

    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2, dw_size: int = 3, stride: int = 1, relu: bool = True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, : self.oup, :, :]


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = out_channels // 4
        self.branch1x1 = nn.Conv2d(in_channels, c, kernel_size=1)
        self.branch3x3 = nn.Sequential(nn.Conv2d(in_channels, c, kernel_size=1), nn.Conv2d(c, c, kernel_size=3, padding=1))
        self.branch5x5 = nn.Sequential(nn.Conv2d(in_channels, c, kernel_size=1), nn.Conv2d(c, c, kernel_size=5, padding=2))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(in_channels, c, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch5x5(x), self.branch_pool(x)], 1)


class ECALayer(nn.Module):
    """Efficient channel attention: a 1-D convolution over the pooled channel vector."""

    def __init__(self, channel: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, -1, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 2):
        super().__init__()
        self.channel_shuffle = ChannelShuffle(groups)
        self.ghost = GhostModule(in_channels, out_channels)
        self.eca = ECALayer(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eca(self.ghost(self.channel_shuffle(x)))


class FeatureFusion(nn.Module):
    """Concatenates the two tracks and gates the channels with a squeeze-and-excitation weight."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((x1, x2), dim=1)
        return fused * self.se(fused)


class DualTrackModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Everything up to and including layer4: a 2048 x 7 x 7 map for a 224 input.
        self.resnet_track = nn.Sequential(*list(resnet.children())[:-2])

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionModule(64, 128)
        self.block1 = ConvBlock(128, 256)
        self.block2 = ConvBlock(256, 512)
        self.final_conv = nn.Conv2d(512, 1024, kernel_size=1, padding=0)
        self.final_bn = nn.BatchNorm2d(1024)

        total_features = 2048 + 1024
        self.feature_fusion = FeatureFusion(total_features, total_features)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(total_features, 1536)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1536, 768)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.resnet_track(x)

        x2 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x2 = self.inception1(x2)
        x2 = self.block1(x2)
        x2 = self.block2(x2)
        x2 = self.relu(self.final_bn(self.final_conv(x2)))

        # The second track is stride 4, the ResNet track stride 32: resample the
        # ResNet map to the second track's size before fusing.
        x1 = nn.functional.adaptive_avg_pool2d(x1, x2.shape[2:])
        fused = self.global_avg_pool(self.feature_fusion(x1, x2))
        fused = torch.flatten(fused, 1)

        x = self.dropout1(self.relu(self.fc1(fused)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(weights: str | None, num_classes: int = NUM_CLASSES, device: torch.device | str = "cpu") -> DualTrackModel:
    """A model ready for inference, from a state_dict file saved by train.py or the notebook."""
    model = DualTrackModel(num_classes, pretrained=weights is None)
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # The notebook saved full training checkpoints every nine epochs and a
        # bare state_dict at the end; accept either.
        model.load_state_dict(state.get("model_state_dict", state))
    return model.to(device).eval()
