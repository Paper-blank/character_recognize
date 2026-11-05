from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from task5.data_utils import (
    CLASS_NAMES_JA,
    CLASS_NAMES_ROMAJI,
    make_train_val_indices,
    prepare_external_image,
)
from task5.models import BaselineMLP, ImprovedCNN, build_model, count_trainable_parameters
from task5.experiment_utils import evaluate_model


@pytest.mark.parametrize("model_class", [BaselineMLP, ImprovedCNN])
def test_model_output_shape(model_class: type[torch.nn.Module]) -> None:
    model = model_class()
    output = model(torch.zeros(4, 1, 28, 28))
    assert output.shape == (4, 10)
    assert count_trainable_parameters(model) > 0


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_model("not-a-model")


def test_label_mapping_has_ten_unique_entries() -> None:
    assert len(CLASS_NAMES_JA) == len(CLASS_NAMES_ROMAJI) == 10
    assert len(set(CLASS_NAMES_JA)) == len(set(CLASS_NAMES_ROMAJI)) == 10


def test_split_is_balanced_and_reproducible() -> None:
    targets = torch.arange(10).repeat_interleave(20)
    train_a, val_a = make_train_val_indices(targets, val_size=50, seed=2026)
    train_b, val_b = make_train_val_indices(targets, val_size=50, seed=2026)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 150
    assert len(val_a) == 50
    assert torch.bincount(targets[val_a]).tolist() == [5] * 10
    assert set(train_a).isdisjoint(val_a)


def test_external_white_background_is_inverted_and_resized(tmp_path: Path) -> None:
    array = np.full((40, 40), 255, dtype=np.uint8)
    array[8:32, 18:22] = 0
    image_path = tmp_path / "character.png"
    Image.fromarray(array).save(image_path)
    tensor, preview = prepare_external_image(image_path)
    assert tensor.shape == (1, 1, 28, 28)
    assert preview.size == (28, 28)
    assert preview.getextrema()[1] > 0


def test_external_small_character_is_enlarged(tmp_path: Path) -> None:
    array = np.zeros((32, 32), dtype=np.uint8)
    array[14:18, 15:17] = 255
    image_path = tmp_path / "small-character.png"
    Image.fromarray(array).save(image_path)
    _, preview = prepare_external_image(image_path)
    foreground = np.asarray(preview) > 20
    rows, columns = np.where(foreground)
    assert rows.max() - rows.min() >= 18
    assert columns.max() - columns.min() >= 8


def test_aligned_28_by_28_image_is_not_rescaled(tmp_path: Path) -> None:
    array = np.zeros((28, 28), dtype=np.uint8)
    array[2:22, 5:8] = 255
    image_path = tmp_path / "aligned.png"
    Image.fromarray(array).save(image_path)
    _, preview = prepare_external_image(image_path)
    np.testing.assert_array_equal(np.asarray(preview), array)


def test_macro_f1_keeps_all_ten_classes_for_limited_data() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    dataset = TensorDataset(torch.zeros(1, 1, 28, 28), torch.zeros(1, dtype=torch.long))
    metrics, _, _ = evaluate_model(
        model,
        DataLoader(dataset, batch_size=1),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(0.1)
