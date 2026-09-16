#!/usr/bin/env python3
"""Minimal inference adapter for MixMo's vanilla CIFAR-100 WRN-28-10.

Architecture and preprocessing are ported from the official MixMo repository:
https://github.com/alexrame/mixmo-pytorch

The official single-network checkpoint returns a dictionary in the original
training framework.  This adapter returns the logits tensor directly so it can
act as the frozen differentiable classifier in HAT-DP/CGPO/PGD/EOT.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# MixMo intentionally uses these shared CIFAR statistics for both CIFAR-10 and
# CIFAR-100 (mixmo/augmentations/standard_augmentations.py).
MIXMO_CIFAR_MEAN = (
    0.4913725490196078,
    0.4823529411764706,
    0.4466666666666667,
)
MIXMO_CIFAR_STD = (0.2023, 0.1994, 0.2010)


class WideBasic(nn.Module):
    """MixMo's pre-activation wide residual block."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=0.1)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.stride = stride
        padding = 1 if stride == 1 else 0
        self.pad1 = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.equalInOut = inplanes == planes
        self.shortcut = nn.Sequential()
        if stride != 1 or not self.equalInOut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(F.relu(self.bn1(inputs)))
        outputs = F.relu(self.bn2(outputs))
        if self.stride != 1 and not self.equalInOut:
            outputs = self.pad1(outputs)
        outputs = self.conv2(outputs)
        return outputs + self.shortcut(inputs)


class MixMoWideResNet(nn.Module):
    """Single-member vanilla MixMo WideResNet, defaulting to WRN-28-10."""

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 4) % 6 == 0")
        blocks_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.depth = depth
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.inplanes = channels[0]

        self.conv1 = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(
            channels[1], blocks=blocks_per_group, stride=1
        )
        self.layer2 = self._make_layer(
            channels[2], blocks=blocks_per_group, stride=2
        )
        self.layer3 = self._make_layer(
            channels[3], blocks=blocks_per_group, stride=2
        )
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(WideBasic(self.inplanes, planes, block_stride))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv1(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = F.relu(self.bn1(features))
        features = self.avgpool(features).flatten(1)
        return self.fc(features)


class MixMoCIFAR100Classifier(nn.Module):
    """Differentiable raw-[0,1] wrapper around the normalized MixMo model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(MIXMO_CIFAR_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(MIXMO_CIFAR_STD).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model((images - self.mean) / self.std)


def _strip_prefixes(
    state: Mapping[str, torch.Tensor], prefixes: Tuple[str, ...] = ("module.",)
) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def extract_mixmo_state(checkpoint: Any) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Extract the network state from official or already-converted checkpoints."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint)!r}")

    state_key = None
    for candidate in (
        "classifier_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        if candidate in checkpoint:
            state_key = candidate
            break

    if state_key is None:
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
            state_key = "raw_state_dict"
        else:
            raise KeyError(
                "Could not find classifier_state_dict/model_state_dict/state_dict "
                f"in checkpoint keys: {sorted(checkpoint.keys())}"
            )
    else:
        state = checkpoint[state_key]

    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint field {state_key!r} is not a state dict")
    metadata = {
        "state_dict_key": state_key,
        "epoch": checkpoint.get("epoch"),
        "checkpoint_keys": sorted(checkpoint.keys()),
    }
    return _strip_prefixes(state), metadata


def load_mixmo_cifar100_classifier(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
) -> tuple[MixMoCIFAR100Classifier, Dict[str, Any]]:
    """Strictly load and structurally validate the official vanilla checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    state, metadata = extract_mixmo_state(payload)
    model = MixMoWideResNet(depth=28, widen_factor=10, num_classes=100)
    model.load_state_dict(state, strict=True)
    classifier = MixMoCIFAR100Classifier(model).to(device).eval()
    classifier.requires_grad_(False)

    if not (
        len(model.layer1) == 4
        and len(model.layer2) == 4
        and len(model.layer3) == 4
        and model.fc.in_features == 640
        and model.fc.out_features == 100
    ):
        raise RuntimeError("Loaded classifier failed WRN-28-10 structural checks")

    with torch.inference_mode():
        output = classifier(torch.zeros(2, 3, 32, 32, device=device))
    if tuple(output.shape) != (2, 100):
        raise RuntimeError(f"Expected classifier output [2,100], got {tuple(output.shape)}")

    metadata.update(
        {
            "source": "alexrame/mixmo-pytorch official vanilla checkpoint",
            "architecture": "WRN-28-10",
            "num_classes": 100,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "reported_top1": 0.8179,
            "input_range": [0.0, 1.0],
            "normalization_mean": MIXMO_CIFAR_MEAN,
            "normalization_std": MIXMO_CIFAR_STD,
        }
    )
    return classifier, metadata
