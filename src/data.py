"""
data.py — Dataset definition, transforms, and data loading for DukeGuessr.

What this file does:
    Provides everything needed to load raw landmark images from disk and
    prepare them for training or evaluation. Handles two separate normalization
    schemes (CLIP vs ImageNet), stratified train/val/test splitting, and a
    configurable augmentation pipeline.

Academic contributions demonstrated here:
    - Stratified train/validation/test split (70/15/15) using sklearn,
      ensuring each class is proportionally represented in all three splits.
    - Four-technique data augmentation pipeline applied during training only:
      random crop, horizontal flip, rotation, and color jitter.
    - Proper input normalization using modality-appropriate statistics —
      CLIP's own pretraining mean/std for CLIP models, ImageNet mean/std
      for the ViT comparison model.
    - PyTorch DataLoader with shuffling, batching, and pin_memory for
      efficient GPU data transfer.
    - Modular design: dataset class, transform factory, split logic, and
      loader factory are all separated so any component can be reused
      independently by training scripts or the evaluation pipeline.

AI attribution: This file was scaffolded with Claude (Anthropic). All design
decisions (normalization choices, augmentation techniques, stratification
parameters) are Alexander Zarboulas's original work. See ATTRIBUTION.md for full details.
"""

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ── Normalization statistics ───────────────────────────────────────────────────
# CLIP was pretrained with its own channel mean/std — using ImageNet stats here
# would create a distribution mismatch between pretraining and fine-tuning.
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Standard ImageNet stats used for the ViT-B/16 comparison model, which was
# pretrained on ImageNet-1K with these normalization values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Folder names must exactly match the subdirectory names inside data/raw/.
# Order matters — index in this list is the integer class label used in training.
CLASS_NAMES = [
    "perkins",
    "main_quad",
    "chapel",
    "bus_stop",
    "gardens",
    "wannamaker_benches",
    "other",
]


# ── Transforms ────────────────────────────────────────────────────────────────
def make_transforms(norm: str = "clip", augment: bool = True):
    """
    Build a torchvision transform pipeline.

    Args:
        norm:    "clip" or "imagenet" — selects the normalization statistics
                 appropriate for the model being trained.
        augment: If True, apply training augmentations (random crop, flip,
                 rotation, color jitter). If False, use a deterministic
                 resize-only pipeline for val/test — augmentation at evaluation
                 time would introduce randomness into reported metrics.

    Returns:
        A transforms.Compose pipeline ready to be passed to DukeLandmarkDataset.

    Augmentation techniques (training only):
        1. RandomCrop(224) after resize to 256 — forces the model to be
           robust to partial views of a landmark.
        2. RandomHorizontalFlip — most landmarks look valid when mirrored,
           doubling effective training data.
        3. RandomRotation(15°) — handles photos taken at a slight tilt.
        4. ColorJitter (brightness/contrast/saturation ±0.2) — simulates
           different lighting and weather conditions across photo sessions.
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if norm == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    normalize = transforms.Normalize(mean=mean, std=std)

    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),           # Technique 1: random crop
            transforms.RandomHorizontalFlip(),    # Technique 2: horizontal flip
            transforms.RandomRotation(15),        # Technique 3: rotation
            transforms.ColorJitter(               # Technique 4: color jitter
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Val/test: deterministic center resize only — no randomness
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class DukeLandmarkDataset(Dataset):
    """
    PyTorch Dataset for Duke landmark images.

    Stores pre-computed file paths and integer labels in memory; images are
    loaded from disk lazily in __getitem__ to avoid RAM bottlenecks with
    large datasets.
    """

    def __init__(self, image_paths: list[str], labels: list[int], transform):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Convert to RGB explicitly — some cameras produce EXIF-rotated or
        # RGBA images; RGB conversion normalizes both cases.
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


# ── Split helpers ─────────────────────────────────────────────────────────────
def collect_paths_and_labels(data_dir: str):
    """
    Walk data/raw/ and collect all image file paths with their class labels.

    Returns:
        paths  : list of absolute path strings
        labels : list of integer class indices (aligned with CLASS_NAMES)
    """
    data_dir = Path(data_dir)
    paths, labels = [], []

    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        # Accept both lower and upper case extensions (cameras vary)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in cls_dir.glob(ext):
                paths.append(str(p))
                labels.append(idx)

    return paths, labels


def make_splits(data_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15,
                seed: int = 42):
    """
    Produce stratified train/val/test splits.

    Stratification (stratify=labels) ensures each class appears in all three
    splits at the same proportion as the full dataset — critical with small
    per-class counts (~50 images) where random splitting could leave a class
    entirely absent from the test set.

    Split ratios: 70% train / 15% val / 15% test.
    The test set is held out completely — it is never used during training or
    hyperparameter selection.

    Args:
        seed: Fixed random seed for reproducibility across runs.

    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    paths, labels = collect_paths_and_labels(data_dir)

    # First split: carve out the test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )

    # Second split: divide remaining data into train and val
    # val_size is recalculated relative to the train+val pool
    val_size = val_ratio / (1 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size, stratify=train_val_labels, random_state=seed
    )

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def make_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 2,
                 norm: str = "clip", seed: int = 42):
    """
    Build train, val, and test DataLoaders in one call.

    Args:
        batch_size:  Number of images per batch. Reduce to 16 if GPU OOM.
        num_workers: Parallel data loading workers. Set to 0 on Windows.
        norm:        "clip" or "imagenet" — passed through to make_transforms.
        seed:        Passed to make_splits for reproducible splitting.

    Returns:
        (train_loader, val_loader, test_loader)

    Notes:
        - Training loader shuffles each epoch so the model never memorizes
          batch order.
        - Val/test loaders do not shuffle — consistent order makes debugging
          and failure analysis easier.
        - pin_memory=True speeds up CPU→GPU transfers on CUDA machines.
    """
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        make_splits(data_dir, seed=seed)

    train_set = DukeLandmarkDataset(train_paths, train_labels,
                                    make_transforms(norm=norm, augment=True))
    val_set   = DukeLandmarkDataset(val_paths,   val_labels,
                                    make_transforms(norm=norm, augment=False))
    test_set  = DukeLandmarkDataset(test_paths,  test_labels,
                                    make_transforms(norm=norm, augment=False))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
