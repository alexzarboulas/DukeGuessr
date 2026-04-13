"""
train_vit.py — Training loop for the ViT-B/16 comparison classifier.

What this file does:
    Trains the ViT-B/16 architecture comparison model using standard
    supervised image classification. This is the conventional alternative
    to CLIP's contrastive approach — images are mapped directly to integer
    class labels via CrossEntropyLoss rather than matched against text
    descriptions.

Academic contributions demonstrated here:
    - Fine-tuning a pretrained vision transformer (ViT-B/16, ImageNet-1K)
      on the 7-class Duke landmark dataset. The full network is updated
      end-to-end including the transformer backbone.
    - Architecture comparison: this training script produces the ViT results
      contrasted against zero-shot CLIP (32.1%) and fine-tuned CLIP (100%)
      in the experiments notebook. Both fine-tuned models reach 100%,
      but the paradigms differ fundamentally.
    - Regularization: early stopping (patience=5) and weight decay (L2,
      default 1e-4) are applied, consistent with the CLIP training setup
      to keep the comparison controlled.
    - Training curves (loss and accuracy per epoch) are saved to JSON for
      visualization in the experiments notebook.
    - ImageNet normalization is used here (vs. CLIP normalization in
      train_clip.py) because ViT-B/16 was pretrained on ImageNet with
      those specific channel statistics.

AI attribution: This file was scaffolded with Claude (Anthropic). Hyperparameter
choices (lr=1e-4), the decision to use ImageNet normalization, and all experiment
design decisions are Alexander Zarboulas's original work. See ATTRIBUTION.md for full details.

Usage:
    python src/train_vit.py --data_dir data/raw --lr 1e-4 --epochs 30 --run_name vit_base
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from data import make_loaders
from vit_model import build_vit


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Halts training when validation loss stops improving.

    Identical implementation to train_clip.py — keeping it local to each
    training script avoids a shared dependency and keeps each script
    self-contained and runnable independently.
    """

    def __init__(self, patience: int = 5):
        self.patience  = patience
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── One epoch of training or validation ───────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Run one full pass over a DataLoader for the ViT classifier.

    Unlike the CLIP training loop, logits here come directly from the model's
    linear head (shape: B x 7) — there is no text encoder or cosine similarity
    step. Loss is standard cross-entropy over class logits.

    Args:
        model:     ViT-B/16 with 7-class head
        loader:    DataLoader yielding (images, labels) with ImageNet normalization
        criterion: nn.CrossEntropyLoss instance
        optimizer: Adam optimizer (only applied when train=True)
        device:    cuda or cpu
        train:     if True, backpropagate and update weights

    Returns:
        (avg_loss, accuracy) for the full loader
    """
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Direct forward pass — no text encoding, no cosine similarity
            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)

    return total_loss / n, correct / n


# ── Main training function ────────────────────────────────────────────────────
def train(
    data_dir:     str,
    lr:           float = 1e-4,
    weight_decay: float = 1e-4,
    epochs:       int   = 30,
    batch_size:   int   = 32,
    patience:     int   = 5,
    save_dir:     str   = "models",
    run_name:     str   = "vit_run",
):
    """
    Fine-tune ViT-B/16 on Duke landmark photos.

    Key design decisions:
        - lr=1e-4: Higher than CLIP's 1e-5 because ViT's task is simpler
          (integer labels vs. embedding alignment) and the head is randomly
          initialized, requiring a larger initial step.
        - norm="imagenet": ViT was pretrained on ImageNet — using CLIP's
          normalization stats here would hurt performance.
        - Full end-to-end fine-tuning: unlike CLIP (frozen text encoder),
          all ViT parameters are trainable including the backbone.
        - Best checkpoint saved by validation accuracy for consistency with
          the CLIP training script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ViT requires ImageNet normalization — different from CLIP's stats
    train_loader, val_loader, _ = make_loaders(data_dir, batch_size=batch_size,
                                               norm="imagenet")

    model     = build_vit().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    stopper      = EarlyStopping(patience=patience)
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    best_path    = os.path.join(save_dir, f"{run_name}_best.pth")

    start = time.time()
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion,
                                  optimizer, device, train=True)
        v_loss, v_acc = run_epoch(model, val_loader,   criterion,
                                  optimizer, device, train=False)

        # Record history for training curve plots in the notebook
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:3d} | train loss {t_loss:.4f} acc {t_acc:.4f} "
              f"| val loss {v_loss:.4f} acc {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  → Saved best model (val acc {best_val_acc:.4f})")

        if stopper.step(v_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min. Best val acc: {best_val_acc:.4f}")

    history_path = os.path.join(save_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    return history, best_val_acc


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="data/raw")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--save_dir",     default="models")
    parser.add_argument("--run_name",     default="vit_run")
    args = parser.parse_args()

    train(
        data_dir     = args.data_dir,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        patience     = args.patience,
        save_dir     = args.save_dir,
        run_name     = args.run_name,
    )
