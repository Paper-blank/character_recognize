"""Compare the saved baseline and improved-model experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .experiment_utils import write_json
except ImportError:
    from experiment_utils import write_json


TASK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=TASK_DIR / "results" / "baseline" / "metrics.json"
    )
    parser.add_argument(
        "--improved", type=Path, default=TASK_DIR / "results" / "improved" / "metrics.json"
    )
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "results")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline = read_json(args.baseline)
    improved = read_json(args.improved)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_accuracy = float(baseline["test_accuracy"])
    improved_accuracy = float(improved["test_accuracy"])
    baseline_f1 = float(baseline["test_macro_f1"])
    improved_f1 = float(improved["test_macro_f1"])
    baseline_error = 1.0 - baseline_accuracy
    improved_error = 1.0 - improved_accuracy
    error_reduction = (baseline_error - improved_error) / baseline_error
    summary = {
        "baseline": baseline,
        "improved": improved,
        "changes": {
            "accuracy_points": (improved_accuracy - baseline_accuracy) * 100,
            "macro_f1_points": (improved_f1 - baseline_f1) * 100,
            "relative_error_reduction": error_reduction,
            "additional_parameters": int(improved["parameter_count"])
            - int(baseline["parameter_count"]),
            "additional_training_seconds": float(improved["training_seconds"])
            - float(baseline["training_seconds"]),
        },
    }
    write_json(args.output_dir / "comparison.json", summary)

    labels = ["Accuracy", "Macro-F1"]
    baseline_values = [baseline_accuracy * 100, baseline_f1 * 100]
    improved_values = [improved_accuracy * 100, improved_f1 * 100]
    positions = [0, 1]
    width = 0.34
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars_baseline = axis.bar(
        [value - width / 2 for value in positions],
        baseline_values,
        width,
        label="Baseline MLP",
        color="#7a9cc6",
    )
    bars_improved = axis.bar(
        [value + width / 2 for value in positions],
        improved_values,
        width,
        label="Improved CNN",
        color="#e07a5f",
    )
    axis.bar_label(bars_baseline, fmt="%.2f%%", padding=3)
    axis.bar_label(bars_improved, fmt="%.2f%%", padding=3)
    axis.set(
        title="KMNIST test-set comparison",
        ylabel="Score (%)",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 100),
    )
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)

    print(
        f"accuracy: {baseline_accuracy:.4f} -> {improved_accuracy:.4f} "
        f"({summary['changes']['accuracy_points']:+.2f} points)"
    )
    print(
        f"macro-F1: {baseline_f1:.4f} -> {improved_f1:.4f} "
        f"({summary['changes']['macro_f1_points']:+.2f} points)"
    )


if __name__ == "__main__":
    main()
