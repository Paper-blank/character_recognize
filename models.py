"""Models used by the KMNIST character-recognition experiment."""

from __future__ import annotations

import torch
from torch import nn


NUM_CLASSES = 10


class BaselineMLP(nn.Module):
    """A small fully connected baseline that ignores 2-D neighbourhoods."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images)


class ImprovedCNN(nn.Module):
    """A compact CNN that keeps local stroke information."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a model by its command-line name."""

    if name == "baseline":
        return BaselineMLP(num_classes=num_classes)
    if name == "improved":
        return ImprovedCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model {name!r}; choose 'baseline' or 'improved'.")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
