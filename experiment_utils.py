"""Training, evaluation, metric, and plotting utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

try:
    from .data_utils import CLASS_NAMES_ROMAJI, KMNIST_MEAN, KMNIST_STD
except ImportError:
    from data_utils import CLASS_NAMES_ROMAJI, KMNIST_MEAN, KMNIST_STD


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    target_array = np.concatenate(all_targets)
    prediction_array = np.concatenate(all_predictions)
    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": float((target_array == prediction_array).mean()),
        "macro_f1": float(
            f1_score(
                target_array,
                prediction_array,
                labels=list(range(len(CLASS_NAMES_ROMAJI))),
                average="macro",
                zero_division=0,
            )
        ),
    }
    return metrics, target_array, prediction_array


def build_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES_ROMAJI))),
        target_names=CLASS_NAMES_ROMAJI,
        output_dict=True,
        zero_division=0,
    )
    return {
        name: {
            key: int(value) if key == "support" else float(value)
            for key, value in report[name].items()
        }
        for name in CLASS_NAMES_ROMAJI
    }


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASS_NAMES_ROMAJI)))
    denominators = matrix.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=np.float64)
    np.divide(matrix, denominators, out=normalized, where=denominators != 0)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set(
        title="Normalized confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(CLASS_NAMES_ROMAJI)),
        yticks=range(len(CLASS_NAMES_ROMAJI)),
        xticklabels=CLASS_NAMES_ROMAJI,
        yticklabels=CLASS_NAMES_ROMAJI,
    )
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_sample_predictions(
    dataset: Any,
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    max_samples: int = 16,
) -> None:
    correct = np.flatnonzero(targets == predictions)
    incorrect = np.flatnonzero(targets != predictions)
    wrong_count = min(max_samples // 2, len(incorrect))
    right_count = min(max_samples - wrong_count, len(correct))
    chosen = np.concatenate([incorrect[:wrong_count], correct[:right_count]])

    columns = 4
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(8, 2.25 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes_array, chosen, strict=False):
        image, _ = dataset[int(index)]
        image_array = image.squeeze(0).numpy() * KMNIST_STD[0] + KMNIST_MEAN[0]
        is_correct = targets[index] == predictions[index]
        axis.imshow(image_array, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"T:{CLASS_NAMES_ROMAJI[targets[index]]}  P:{CLASS_NAMES_ROMAJI[predictions[index]]}",
            color="green" if is_correct else "crimson",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(chosen) :]:
        axis.axis("off")
    figure.suptitle("Test predictions (red = wrong, green = correct)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
