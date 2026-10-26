"""
Classify one or more images with saved weights.

    python -m indfood.predict --weights runs/baseline/best.pth photo.jpg another.jpg

Prints the top guesses for each image. The class names come from the dataset
folder (or `--classes` pointing at a JSON list) so the output reads as dish
names rather than indices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import eval_transform
from .model import load_model


def class_names(args: argparse.Namespace) -> list[str]:
    if args.classes:
        return json.load(open(args.classes))
    train = Path(args.data) / "train"
    return sorted(p.name for p in train.iterdir() if p.is_dir())


@torch.no_grad()
def predict(model, images: list[str], classes: list[str], device, image_size: int = 224, top: int = 3) -> list[list[tuple[str, float]]]:
    tf = eval_transform(image_size)
    batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in images]).to(device)
    probs = model(batch).softmax(1).cpu()
    results = []
    for row in probs:
        values, indices = row.topk(min(top, len(classes)))
        results.append([(classes[i], float(v)) for v, i in zip(values, indices)])
    return results


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    classes = class_names(args)
    model = load_model(args.weights, len(classes), device)
    for path, guesses in zip(args.images, predict(model, args.images, classes, device, args.image_size, args.top)):
        best, p = guesses[0]
        rest = ", ".join(f"{name} {prob:.0%}" for name, prob in guesses[1:])
        print(f"{path}: {best} ({p:.0%})" + (f"; then {rest}" if rest else ""))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="ind_food_37", help="used only for the class names")
    p.add_argument("--classes", help="JSON list of class names, instead of reading the dataset folder")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
