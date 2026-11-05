"""Train and evaluate a baseline MLP or an improved CNN on KMNIST."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    from .data_utils import (
        CLASS_NAMES_JA,
        CLASS_NAMES_ROMAJI,
        KMNIST_MEAN,
        KMNIST_STD,
        load_kmnist,
        make_train_val_indices,
    )
    from .experiment_utils import (
        build_classification_metrics,
        evaluate_model,
        resolve_device,
        save_confusion_matrix,
        save_sample_predictions,
        save_training_curves,
        set_seed,
        train_one_epoch,
        write_json,
    )
    from .models import build_model, count_trainable_parameters
except ImportError:
    from data_utils import (
        CLASS_NAMES_JA,
        CLASS_NAMES_ROMAJI,
        KMNIST_MEAN,
        KMNIST_STD,
        load_kmnist,
        make_train_val_indices,
    )
    from experiment_utils import (
        build_classification_metrics,
        evaluate_model,
        resolve_device,
        save_confusion_matrix,
        save_sample_predictions,
        save_training_curves,
        set_seed,
        train_one_epoch,
        write_json,
    )
    from models import build_model, count_trainable_parameters


TASK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "improved"), required=True)
    parser.add_argument("--data-dir", type=Path, default=TASK_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-size", type=int, default=6000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def make_loader(
    dataset: object,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir or TASK_DIR / "results" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    use_augmentation = args.model == "improved"
    training_dataset = load_kmnist(args.data_dir, train=True, augment=use_augmentation)
    validation_dataset = load_kmnist(args.data_dir, train=True, augment=False)
    test_dataset_full = load_kmnist(args.data_dir, train=False, augment=False)
    train_indices, val_indices = make_train_val_indices(
        training_dataset.targets,
        val_size=args.val_size,
        seed=args.seed,
    )
    if args.train_limit is not None:
        train_indices = train_indices[: args.train_limit]
    if args.val_limit is not None:
        val_indices = val_indices[: args.val_limit]
    test_indices = list(range(len(test_dataset_full)))
    if args.test_limit is not None:
        test_indices = test_indices[: args.test_limit]

    train_dataset = Subset(training_dataset, train_indices)
    val_dataset = Subset(validation_dataset, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device)

    model = build_model(args.model).to(device)
    parameter_count = count_trainable_parameters(model)
    label_smoothing = 0.05 if args.model == "improved" else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if args.model == "improved":
        optimizer_name = "AdamW"
        scheduler_name: str | None = "CosineAnnealingLR"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        )
    else:
        optimizer_name = "Adam"
        scheduler_name = None
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    training_config = {
        "augmentation": use_augmentation,
        "label_smoothing": label_smoothing,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "validation_size_requested": args.val_size,
        "normalization_source": "official_60000_image_training_split",
    }

    print(
        f"model={args.model} device={device} parameters={parameter_count:,} "
        f"train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
    )
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "best.pt"
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                    "class_names_ja": CLASS_NAMES_JA,
                    "class_names_romaji": CLASS_NAMES_ROMAJI,
                    "normalization": {"mean": KMNIST_MEAN, "std": KMNIST_STD},
                    "training_config": training_config,
                },
                checkpoint_path,
            )
        if scheduler is not None:
            scheduler.step()

    training_seconds = time.perf_counter() - started_at
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_targets, test_predictions = evaluate_model(
        model, test_loader, criterion, device
    )

    metrics = {
        "dataset": "KMNIST",
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_config": training_config,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "per_class": build_classification_metrics(test_targets, test_predictions),
    }
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "metrics.json", metrics)
    save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix(test_targets, test_predictions, output_dir / "confusion_matrix.png")
    save_sample_predictions(
        test_dataset,
        test_targets,
        test_predictions,
        output_dir / "sample_predictions.png",
    )
    print(
        f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"best_epoch={best_epoch} time={training_seconds:.1f}s output={output_dir}"
    )


if __name__ == "__main__":
    main()
