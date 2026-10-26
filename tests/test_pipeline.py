"""
Runs the whole pipeline, train then evaluate then predict, on a tiny synthetic
ImageFolder with random ResNet weights, so it finishes on a CPU in well under a
minute and never downloads anything. It proves the plumbing, not the accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from indfood import evaluate, predict, train
from indfood.model import DualTrackModel, count_parameters, load_model

CLASSES = ("Dosa", "Jalebi", "Samosa")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ind_food_tiny")
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        for cls in CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(folder / f"{i}.jpg")
    return root


def test_model_shape_and_size():
    model = DualTrackModel(37, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 37)
    # The README's figure. The trunk is ResNet-50 but the whole network is half again as large.
    assert count_parameters(model) == 39_544_363


def test_train_evaluate_predict(tiny_dataset: Path, tmp_path: Path):
    out = tmp_path / "run"
    args = train.parser().parse_args([
        "--data", str(tiny_dataset), "--out", str(out), "--epochs", "2", "--batch-size", "4",
        "--image-size", "48", "--no-pretrained", "--cpu", "--augment",
    ])
    history = train.train(args)

    assert [h["epoch"] for h in history] == [0, 1]
    assert set(history[0]) >= {"train_loss", "train_acc", "val_loss", "val_acc"}
    assert json.load(open(out / "history.json")) == history
    assert json.load(open(out / "config.json"))["classes"] == list(CLASSES)
    assert (out / "best.pth").exists()

    eval_args = evaluate.parser().parse_args([
        "--weights", str(out / "best.pth"), "--data", str(tiny_dataset), "--split", "test",
        "--out", str(tmp_path / "eval"), "--image-size", "48", "--cpu",
    ])
    evaluate.main(eval_args)
    rows = json.load(open(tmp_path / "eval" / "per_class_accuracy.json"))
    assert [r["class"] for r in rows] == list(CLASSES)
    assert all(0.0 <= r["accuracy"] <= 1.0 and r["images"] == 2 for r in rows)
    matrix = json.load(open(tmp_path / "eval" / "confusion_matrix.json"))["matrix"]
    assert sum(map(sum, matrix)) == 6

    model = load_model(str(out / "best.pth"), len(CLASSES))
    image = next((tiny_dataset / "test" / "Dosa").glob("*.jpg"))
    guesses = predict.predict(model, [str(image)], list(CLASSES), "cpu", image_size=48, top=2)
    assert len(guesses) == 1 and len(guesses[0]) == 2
    assert {name for name, _ in guesses[0]} <= set(CLASSES)
    assert guesses[0][0][1] >= guesses[0][1][1]


def test_loads_a_notebook_style_checkpoint(tmp_path: Path):
    """The notebook saved {'model_state_dict': ...} every nine epochs; those load too."""
    model = DualTrackModel(3, pretrained=False)
    torch.save({"epoch": 9, "model_state_dict": model.state_dict()}, tmp_path / "ckpt.pt")
    loaded = load_model(str(tmp_path / "ckpt.pt"), 3)
    assert torch.equal(loaded.fc3.weight, model.fc3.weight)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
