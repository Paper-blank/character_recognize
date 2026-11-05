"""Recognize one external character image with a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .data_utils import CLASS_NAMES_JA, CLASS_NAMES_ROMAJI, prepare_external_image
    from .experiment_utils import resolve_device
    from .models import build_model
except ImportError:
    from data_utils import CLASS_NAMES_JA, CLASS_NAMES_ROMAJI, prepare_external_image
    from experiment_utils import resolve_device
    from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.top_k <= len(CLASS_NAMES_JA):
        raise ValueError(f"top-k must be between 1 and {len(CLASS_NAMES_JA)}")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(str(checkpoint["model_name"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor, _ = prepare_external_image(args.image)
    with torch.inference_mode():
        probabilities = model(image_tensor.to(device)).softmax(dim=1)[0]
    scores, indices = probabilities.topk(args.top_k)
    predictions = [
        {
            "label_id": int(index),
            "character": CLASS_NAMES_JA[int(index)],
            "romaji": CLASS_NAMES_ROMAJI[int(index)],
            "probability": float(score),
        }
        for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
    ]
    print(json.dumps({"image": str(args.image), "predictions": predictions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
