"""
Dataset loading. ind_food_37 is in ImageFolder layout, one folder per class
under train/, val/ and test/, so torchvision does the reading.

The published model was trained with no augmentation at all: resize to 224 and
normalise with ImageNet statistics. `augment=True` adds the standard set for a
fine-tuning run (random resized crop, horizontal flip, mild colour jitter) and
is what the augmentation experiment in results/ compares against the baseline.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLITS = ("train", "val", "test")


def eval_transform(image_size: int = 224) -> transforms.Compose:
    """Exactly what the notebook used for every split."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    if not augment:
        return eval_transform(image_size)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def dataset(root: str | Path, split: str, image_size: int = 224, augment: bool = False) -> datasets.ImageFolder:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    tf = train_transform(image_size, augment) if split == "train" else eval_transform(image_size)
    return datasets.ImageFolder(Path(root) / split, transform=tf)


def loader(ds: datasets.ImageFolder, batch_size: int, shuffle: bool, workers: int = 0) -> DataLoader:
    cuda = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=cuda, persistent_workers=workers > 0)
