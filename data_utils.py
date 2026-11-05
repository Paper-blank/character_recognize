"""Dataset and image-preprocessing helpers for KMNIST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.datasets import KMNIST


# The label order is defined by the official KMNIST mapping file.
CLASS_NAMES_JA = ["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"]
CLASS_NAMES_ROMAJI = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

# Statistics computed from the 60,000-image KMNIST training split.
KMNIST_MEAN = (0.1918,)
KMNIST_STD = (0.3483,)


def make_transform(augment: bool = False) -> transforms.Compose:
    operations: list[object] = []
    if augment:
        operations.append(
            transforms.RandomAffine(
                degrees=8,
                translate=(0.08, 0.08),
                scale=(0.92, 1.08),
                fill=0,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(KMNIST_MEAN, KMNIST_STD),
        ]
    )
    return transforms.Compose(operations)


def load_kmnist(
    data_dir: str | Path,
    train: bool,
    augment: bool = False,
    download: bool = True,
) -> KMNIST:
    return KMNIST(
        root=str(data_dir),
        train=train,
        transform=make_transform(augment=augment),
        download=download,
    )


def make_train_val_indices(
    targets: torch.Tensor,
    val_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic class-balanced train/validation split."""

    if val_size <= 0 or val_size >= len(targets):
        raise ValueError("val_size must be between 1 and len(targets) - 1")

    rng = np.random.default_rng(seed)
    targets_array = targets.cpu().numpy()
    class_ids = np.unique(targets_array)
    base_count, remainder = divmod(val_size, len(class_ids))
    train_indices: list[int] = []
    val_indices: list[int] = []

    for offset, class_id in enumerate(class_ids):
        indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(indices)
        class_val_size = base_count + int(offset < remainder)
        val_indices.extend(indices[:class_val_size].tolist())
        train_indices.extend(indices[class_val_size:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def prepare_external_image(path: str | Path) -> tuple[torch.Tensor, Image.Image]:
    """Convert a user image into a centred KMNIST-style 28x28 tensor."""

    image = Image.open(path).convert("L")
    array = np.asarray(image)
    if float(array.mean()) > 127.0:
        image = ImageOps.invert(image)

    # A 28x28 input is treated as already aligned, matching the training pipeline.
    if image.size == (28, 28):
        canvas = image
    else:
        bbox = image.getbbox()
        if bbox is not None:
            image = image.crop(bbox)
        scale = min(20 / max(image.width, 1), 20 / max(image.height, 1))
        resized_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(resized_size, Image.Resampling.LANCZOS)

        canvas = Image.new("L", (28, 28), color=0)
        left = (28 - image.width) // 2
        top = (28 - image.height) // 2
        canvas.paste(image, (left, top))
    tensor = make_transform(augment=False)(canvas).unsqueeze(0)
    return tensor, canvas
