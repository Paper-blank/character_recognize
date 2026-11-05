"""Re-evaluate a saved KMNIST checkpoint on the official test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .data_utils import load_kmnist
    from .experiment_utils import (
        build_classification_metrics,
        evaluate_model,
        resolve_device,
        save_confusion_matrix,
        save_sample_predictions,
        write_json,
    )
    from .models import build_model
except ImportError:
    from data_utils import load_kmnist
    from experiment_utils import (
        build_classification_metrics,
        evaluate_model,
        resolve_device,
        save_confusion_matrix,
        save_sample_predictions,
        write_json,
    )
    from models import build_model


TASK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=TASK_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir or args.checkpoint.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = str(checkpoint["model_name"])
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = load_kmnist(args.data_dir, train=False, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    default_smoothing = 0.05 if model_name == "improved" else 0.0
    training_config = dict(checkpoint.get("training_config", {}))
    training_config.setdefault("label_smoothing", default_smoothing)
    label_smoothing = float(training_config["label_smoothing"])
    metrics, targets, predictions = evaluate_model(
        model,
        loader,
        nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        device,
    )
    payload = {
        "dataset": "KMNIST official test split",
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "label_smoothing": label_smoothing,
        "training_config": training_config,
        "test_samples": len(dataset),
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "per_class": build_classification_metrics(targets, predictions),
    }
    write_json(output_dir / "evaluation.json", payload)
    save_confusion_matrix(targets, predictions, output_dir / "confusion_matrix.png")
    save_sample_predictions(dataset, targets, predictions, output_dir / "sample_predictions.png")
    print(
        f"model={model_name} test_acc={metrics['accuracy']:.4f} "
        f"test_macro_f1={metrics['macro_f1']:.4f} output={output_dir}"
    )


if __name__ == "__main__":
    main()
