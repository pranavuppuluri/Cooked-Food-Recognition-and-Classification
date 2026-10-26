"""
Score saved weights on a split and write the per-class numbers the README charts.

    python -m indfood.evaluate --weights runs/baseline/best.pth --data ind_food_37 --split test --out results

Writes <out>/per_class_accuracy.json in the shape results/make_charts.py reads,
plus <out>/confusion_matrix.json (rows are the true class, columns the
prediction, in class order) so the confusions behind a weak class can be read
off rather than guessed at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import dataset, loader
from .model import load_model


@torch.no_grad()
def evaluate(model, batches, num_classes: int, device) -> tuple[list[list[int]], float]:
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for inputs, labels in batches:
        preds = model(inputs.to(device)).argmax(1).cpu()
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t][p] += 1
    correct = sum(confusion[i][i] for i in range(num_classes))
    return confusion, correct / sum(map(sum, confusion))


def per_class(confusion: list[list[int]], classes: list[str]) -> list[dict]:
    rows = []
    for i, name in enumerate(classes):
        total = sum(confusion[i])
        rows.append({"class": name, "accuracy": round(confusion[i][i] / total, 4) if total else 0.0, "images": total})
    return rows


def top_confusions(confusion: list[list[int]], classes: list[str], n: int = 10) -> list[tuple[str, str, int]]:
    pairs = [
        (classes[i], classes[j], confusion[i][j])
        for i in range(len(classes)) for j in range(len(classes)) if i != j and confusion[i][j]
    ]
    return sorted(pairs, key=lambda p: -p[2])[:n]


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = dataset(args.data, args.split, args.image_size)
    model = load_model(args.weights, len(ds.classes), device)
    confusion, accuracy = evaluate(model, loader(ds, args.batch_size, shuffle=False, workers=args.workers), len(ds.classes), device)

    rows = per_class(confusion, ds.classes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "per_class_accuracy.json", "w"), indent=2)
    json.dump({"classes": ds.classes, "matrix": confusion}, open(out / "confusion_matrix.json", "w"), indent=2)

    mean = sum(r["accuracy"] for r in rows) / len(rows)
    print(f"{args.split}: {len(ds)} images, accuracy {accuracy:.4f}, mean per-class accuracy {mean:.4f}")
    print("most confused pairs (true -> predicted):")
    for t, p, n in top_confusions(confusion, ds.classes):
        print(f"  {n:3d}  {t} -> {p}")
    print(f"wrote {out / 'per_class_accuracy.json'} and {out / 'confusion_matrix.json'}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="a state_dict saved by train.py or the notebook")
    p.add_argument("--data", default="ind_food_37")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--out", default="results/eval")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
