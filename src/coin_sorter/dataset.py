"""PyTorch dataset for the coin sorter, used on Colab.

The on-disk layout matches what ``capture.py`` produces and what Ultralytics
classification training expects::

    data/raw/
        penny_lincoln_obv/
            <timestamp>.jpg
            ...
        penny_wheat_rev/
        nickel_jefferson_obv/
        ...
        reject/

Folder names are the (denomination, design, side) labels from
``classifier.labels`` in config.yaml; ``_rejected/`` is ignored.

This module is consumed only on the training side (Colab / workstation).
It deliberately does ``import torch`` at function scope so the Pi never
needs torch installed just to import the package.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.utils.data import Dataset

log = logging.getLogger("coin_sorter.dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_classes(root: Path) -> list[str]:
    """Return class names = immediate subdirectories of `root`, sorted.

    Sorted alphabetically because that is how Ultralytics assigns class indices,
    and we want training-time and inference-time indices to agree.
    """
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if not classes:
        raise ValueError(f"No class subdirectories found under {root}")
    return classes


def discover_samples(root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    """Return [(image_path, class_index), ...] across all classes."""
    samples: list[tuple[Path, int]] = []
    for idx, name in enumerate(classes):
        cls_dir = root / name
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                samples.append((p, idx))
    if not samples:
        raise ValueError(f"No images found under {root}")
    return samples


def split_train_val(
    samples: list[tuple[Path, int]],
    train_frac: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Deterministic train/val split, stratified per class."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for path, idx in samples:
        by_class.setdefault(idx, []).append((path, idx))

    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for idx, items in by_class.items():
        rng.shuffle(items)
        cut = max(1, int(len(items) * train_frac))
        train.extend(items[:cut])
        val.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_transforms(imgsz: int, train: bool) -> Any:
    """Build torchvision transforms appropriate for coins on a belt.

    Coins land in *any* orientation, so full 360° rotation is non-optional.
    HSV jitter helps with copper-vs-silver tone shift under varying light;
    a small affine shear/scale covers the gantry-height tolerance.
    """
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((imgsz + 32, imgsz + 32)),
                transforms.RandomRotation(degrees=180, expand=False),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=2
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
                transforms.CenterCrop(imgsz),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ]
    )


class CoinDataset:
    """`torch.utils.data.Dataset` over a folder-per-class layout.

    Defined as a plain class with a `_dataset_init` indirection so that
    ``import coin_sorter.dataset`` works on the Pi (where torch is absent).
    The torch base class is mixed in lazily via :meth:`as_torch_dataset`.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        classes: list[str],
        imgsz: int,
        train: bool,
    ) -> None:
        self.samples = samples
        self.classes = classes
        self.imgsz = imgsz
        self.train = train
        self._transform = build_transforms(imgsz, train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        from PIL import Image  # local import — Pi does not need PIL

        path, label = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self._transform(img)
        return tensor, label

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        imgsz: int,
        train_frac: float,
        seed: int,
    ) -> tuple["CoinDataset", "CoinDataset", list[str]]:
        """Build (train_ds, val_ds, classes) from a folder-per-class root."""
        root_p = Path(root)
        classes = discover_classes(root_p)
        samples = discover_samples(root_p, classes)
        train_s, val_s = split_train_val(samples, train_frac, seed)
        log.info(
            "Discovered %d samples across %d classes (train=%d, val=%d)",
            len(samples),
            len(classes),
            len(train_s),
            len(val_s),
        )
        return (
            cls(train_s, classes, imgsz, train=True),
            cls(val_s, classes, imgsz, train=False),
            classes,
        )
