"""
Train the dual-track model on ind_food_37.

    python -m indfood.train --data ind_food_37 --epochs 80 --out runs/baseline
    python -m indfood.train --data ind_food_37 --epochs 80 --out runs/augment --augment

The recipe is the notebook's: SGD at 1e-3 with momentum 0.9, cross-entropy,
ReduceLROnPlateau on validation loss (factor 0.2, patience 3), gradients
clipped to norm 1. Every epoch appends to <out>/history.json in the same shape
as results/training_history.json, and the weights with the best validation
accuracy so far are kept at <out>/best.pth.

`--limit N` trains on the first N images of each split, which is how the
smoke test runs the whole loop on a CPU in seconds.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .data import dataset, loader
from .model import DualTrackModel, count_parameters


def run_epoch(model, batches, criterion, device, optimizer=None, scaler=None):
    """One pass over `batches`. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in batches:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if training:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += inputs.size(0)
    return total_loss / seen, correct / seen


def train(args: argparse.Namespace) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = dataset(args.data, "train", args.image_size, augment=args.augment)
    val_ds = dataset(args.data, "val", args.image_size)
    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))
    classes = (train_ds.dataset if args.limit else train_ds).classes
    train_batches = loader(train_ds, args.batch_size, shuffle=True, workers=args.workers)
    val_batches = loader(val_ds, args.batch_size, shuffle=False, workers=args.workers)

    model = DualTrackModel(len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler() if device.type == "cuda" and args.amp else None

    json.dump(
        {**vars(args), "device": str(device), "classes": classes, "trainable_parameters": count_parameters(model)},
        open(out / "config.json", "w"), indent=2,
    )
    print(f"{device}, {len(train_ds)} train / {len(val_ds)} val images, {count_parameters(model):,} trainable parameters")

    history: list[dict] = []
    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        started = time.time()
        train_loss, train_acc = run_epoch(model, train_batches, criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_batches, criterion, device)
        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
            torch.save(best_state, out / "best.pth")
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - started, 1),
        })
        json.dump(history, open(out / "history.json", "w"), indent=2)
        print(
            f"epoch {epoch:3d}  train {train_loss:.4f} / {train_acc:.4f}   val {val_loss:.4f} / {val_acc:.4f}"
            f"   best {best_acc:.4f}   {history[-1]['seconds']:.0f}s"
        )

    print(f"best validation accuracy {best_acc:.4f}; weights in {out / 'best.pth'}")
    return history


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="ind_food_37", help="dataset root with train/, val/ and test/")
    p.add_argument("--out", default="runs/latest", help="where history.json, config.json and best.pth go")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--augment", action="store_true", help="random crop, flip and colour jitter on the training split")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="use only the first N images of each split")
    p.add_argument("--no-pretrained", action="store_true", help="random ResNet weights; for tests, not for results")
    p.add_argument("--cpu", action="store_true", help="ignore any GPU")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
