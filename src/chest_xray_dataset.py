"""
src/chest_xray_dataset.py
==========================
Dataset utilities for the Chest X-Ray Pneumonia baseline (Phase 6).

Responsibilities
----------------
- Build a stratified train / validation split from the *training* folder.
- Return PyTorch Dataset objects for all three sets (train, val, test).
- Apply training-only augmentation; val and test see only resize + normalize.
- Compute class weights for weighted loss.

The test set is NEVER touched except for final evaluation.
IU X-ray files are NEVER modified or accessed by this module.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

CLASSES = ("NORMAL", "PNEUMONIA")
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}

# ImageNet normalisation statistics (standard for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224   # pixels


# -------------------------------------------------------------------------
# Transform factories
# -------------------------------------------------------------------------

def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Augmentation pipeline for the *training* split only.

    Includes random horizontal flip, small random rotation,
    colour jitter (brightness / contrast), and ImageNet normalisation.
    All images are converted to RGB and resized to image_size × image_size.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Minimal pipeline for validation and test sets.  No augmentation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -------------------------------------------------------------------------
# File listing
# -------------------------------------------------------------------------

def collect_image_paths(folder: Path) -> list[dict]:
    """
    Collect all .jpeg paths from folder/NORMAL and folder/PNEUMONIA.

    Parameters
    ----------
    folder : Path
        A split root that has NORMAL/ and PNEUMONIA/ sub-folders.

    Returns
    -------
    list[dict]
        Each dict: {"path": Path, "label": int, "class_name": str}
    """
    records: list[dict] = []
    for cls_name, label in CLASS_TO_IDX.items():
        cls_dir = folder / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder not found: {cls_dir}"
            )
        for img_path in sorted(cls_dir.glob("*.jpeg")):
            records.append({
                "path": img_path,
                "label": label,
                "class_name": cls_name,
            })
    return records


# -------------------------------------------------------------------------
# Stratified split
# -------------------------------------------------------------------------

def stratified_split(
    records: list[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split `records` into train and validation subsets, preserving the
    class distribution.

    Parameters
    ----------
    records      : list[dict]  from collect_image_paths()
    val_fraction : float       fraction of records to put in validation (0 < f < 1)
    seed         : int         random seed for reproducibility

    Returns
    -------
    (train_records, val_records)
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    rng = random.Random(seed)

    # Separate by class
    by_class: dict[int, list[dict]] = {}
    for rec in records:
        by_class.setdefault(rec["label"], []).append(rec)

    train_recs: list[dict] = []
    val_recs:   list[dict] = []

    for label, class_records in sorted(by_class.items()):
        shuffled = class_records[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_recs.extend(shuffled[:n_val])
        train_recs.extend(shuffled[n_val:])

    # Shuffle the final lists so batches see mixed classes
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)

    return train_recs, val_recs


# -------------------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------------------

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the Chest X-Ray Pneumonia images.

    Parameters
    ----------
    records   : list[dict]        output of collect_image_paths() or a split slice
    transform : transforms.Compose  applied to each image
    """

    def __init__(self, records: list[dict], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img = self.transform(img)
        return img, rec["label"]

    @property
    def labels(self) -> list[int]:
        """Flat list of integer labels — used for class-weight computation."""
        return [r["label"] for r in self.records]


# -------------------------------------------------------------------------
# Class-weight computation
# -------------------------------------------------------------------------

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use with
    ``torch.nn.CrossEntropyLoss(weight=...)``.

    A class that appears less frequently gets a higher weight so the model
    pays more attention to it during training.

    Parameters
    ----------
    labels : list[int]
        Integer class labels (0 = NORMAL, 1 = PNEUMONIA).

    Returns
    -------
    torch.Tensor of shape (num_classes,)
    """
    from collections import Counter
    counts = Counter(labels)
    n_total = len(labels)
    n_classes = len(CLASSES)

    weights = torch.zeros(n_classes, dtype=torch.float32)
    for cls_idx in range(n_classes):
        n_cls = counts.get(cls_idx, 1)
        # Weight = total_samples / (n_classes * count_for_this_class)
        weights[cls_idx] = n_total / (n_classes * n_cls)

    return weights


# -------------------------------------------------------------------------
# Dataset factory — one-stop shop
# -------------------------------------------------------------------------

def build_datasets(
    dataset_root: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """
    Build train, val, and test datasets from the Chest X-Ray Pneumonia folder.

    The val set is carved out of the training folder using a stratified split.
    The original ``val/`` folder (only 16 images) is NOT used.
    The ``test/`` folder is left for final evaluation only.

    Parameters
    ----------
    dataset_root : Path  root containing train/, val/, test/
    val_fraction : float fraction of training images to use as val
    seed         : int   random seed
    image_size   : int   pixels (square resize target)

    Returns
    -------
    dict with keys:
        train_dataset, val_dataset, test_dataset  → ChestXrayDataset
        train_records, val_records, test_records  → list[dict]
        class_weights                             → torch.Tensor
        split_counts                              → dict
    """
    train_folder = dataset_root / "train"
    test_folder  = dataset_root / "test"

    # Collect all training images
    all_train = collect_image_paths(train_folder)

    # Stratified split
    train_recs, val_recs = stratified_split(all_train, val_fraction, seed)

    # Test records
    test_recs = collect_image_paths(test_folder)

    # Build datasets
    train_ds = ChestXrayDataset(train_recs, get_train_transforms(image_size))
    val_ds   = ChestXrayDataset(val_recs,   get_eval_transforms(image_size))
    test_ds  = ChestXrayDataset(test_recs,  get_eval_transforms(image_size))

    # Class weights from training labels only
    weights = compute_class_weights(train_ds.labels)

    # Count summary
    from collections import Counter
    def _counts(recs: list[dict]) -> dict[str, int]:
        c = Counter(r["class_name"] for r in recs)
        return {"NORMAL": c.get("NORMAL", 0), "PNEUMONIA": c.get("PNEUMONIA", 0),
                "total": len(recs)}

    split_counts = {
        "train": _counts(train_recs),
        "val":   _counts(val_recs),
        "test":  _counts(test_recs),
    }

    return {
        "train_dataset": train_ds,
        "val_dataset":   val_ds,
        "test_dataset":  test_ds,
        "train_records": train_recs,
        "val_records":   val_recs,
        "test_records":  test_recs,
        "class_weights": weights,
        "split_counts":  split_counts,
    }
