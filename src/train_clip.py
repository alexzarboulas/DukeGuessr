"""
train_clip.py — Fine-tuning loop for CLIP's image encoder on Duke landmark photos.

What this file does:
    Trains the primary DukeGuessr model by fine-tuning CLIP's image encoder
    to align Duke campus photos with their natural language landmark descriptions.
    The text encoder is frozen throughout — only the image encoder learns.
    Logs training and validation loss/accuracy each epoch, applies early stopping,
    and saves the best checkpoint by validation accuracy.

Academic contributions demonstrated here:
    - Fine-tuning a pretrained vision-language model (CLIP ViT-B/32) on a
      custom dataset. The image encoder is adapted to a new visual domain
      (Duke campus photography) while reusing CLIP's pretrained text
      representations as fixed class anchors.
    - Regularization via two complementary techniques:
        1. Early stopping (patience=5) — halts training when validation loss
           stops improving, preventing overfitting to the small training set.
        2. Weight decay (L2 penalty, default 1e-4) — applied via Adam's
           weight_decay argument, penalizing large weight magnitudes.
    - Systematic hyperparameter tuning: this script supports --lr and --run_name
      flags used to run the LR sweep (1e-4, 1e-5, 1e-6) documented in the
      experiments notebook.
    - Ablation support: --use_short_labels flag enables Ablation A (short
      label strings vs. rich paragraph descriptions as text anchors).
    - Training history (loss and accuracy per epoch) is saved as JSON for
      plotting training curves in the experiments notebook.
    - Documented improvement iterations: the iteration log in the notebook
      was built by running this script at different configurations and
      recording what changed and why.

AI attribution: This file was scaffolded with Claude (Anthropic). All
hyperparameter choices (lr=1e-5, patience=5, weight_decay=1e-4), the experiment
design (LR sweep, short-label ablation), and the iteration log are Alexander
Zarboulas's original work. See ATTRIBUTION.md for full details.

Usage:
    python src/train_clip.py --data_dir data/raw \
                             --descriptions data/descriptions/landmarks.json \
                             --lr 1e-5 --epochs 30 --run_name clip_base
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import make_loaders
from clip_model import (
    load_clip, freeze_text_encoder, encode_descriptions,
    encode_short_labels, compute_logits
)


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Monitors validation loss and signals when training should stop.

    Stops training if val_loss has not improved for `patience` consecutive
    epochs. This prevents overfitting on the small ~245-image training set
    (70% of 350 total) and saves compute time — both CLIP and ViT converged
    well before 30 epochs in practice.
    """

    def __init__(self, patience: int = 5):
        self.patience  = patience
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        """
        Returns True if training should stop, False otherwise.
        Resets the counter whenever a new best loss is found.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return False  # improvement found — keep training
        self.counter += 1
        return self.counter >= self.patience  # no improvement for `patience` epochs


# ── One epoch of training or validation ───────────────────────────────────────
def run_epoch(model, loader, text_features, optimizer, device, train: bool):
    """
    Run one full pass over a DataLoader, computing loss and accuracy.

    Args:
        model:         CLIP model with image encoder trainable
        loader:        DataLoader yielding (images, labels) batches
        text_features: (7, 512) pre-encoded, normalized text anchor embeddings
        optimizer:     Adam optimizer (only used when train=True)
        device:        cuda or cpu
        train:         if True, compute gradients and update weights;
                       if False, run in evaluation mode (no grad, no update)

    Returns:
        (avg_loss, accuracy) over the full loader
    """
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Compute cosine similarity logits between image embeddings
            # and the 7 pre-encoded text anchor embeddings
            logits = compute_logits(model, images, text_features)

            # Cross-entropy over the similarity scores acts as contrastive loss:
            # the model is rewarded for making the correct text anchor most similar
            loss = F.cross_entropy(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)

    return total_loss / n, correct / n


# ── Zero-shot evaluation ──────────────────────────────────────────────────────
def zero_shot_eval(model, loader, text_features, device):
    """
    Evaluate CLIP with no fine-tuning — purely pretrained zero-shot performance.

    This establishes the baseline: how well does vanilla CLIP (pretrained on
    400M internet image-text pairs) classify Duke campus photos without ever
    seeing any of them during training?

    Result: 32.1% test accuracy with paragraph anchors, 49.1% with short labels.
    Both are significantly below the fine-tuned model's 100%.
    """
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = compute_logits(model, images, text_features)
            correct += (logits.argmax(1) == labels).sum().item()
            n       += len(labels)
    return correct / n


# ── Main training function ────────────────────────────────────────────────────
def train(
    data_dir:        str,
    descriptions_path: str,
    lr:              float = 1e-5,
    weight_decay:    float = 1e-4,
    epochs:          int   = 30,
    batch_size:      int   = 32,
    patience:        int   = 5,
    save_dir:        str   = "models",
    run_name:        str   = "clip_run",
    use_short_labels: bool = False,
    clip_model_name: str   = "ViT-B/32",
):
    """
    Fine-tune CLIP's image encoder on Duke landmark photos.

    Key design decisions:
        - lr=1e-5: Low learning rate preserves CLIP's pretrained representations.
          The LR sweep showed 1e-4 destroys pretrained features (39.6% accuracy)
          and 1e-6 converges too slowly (96.2% in 20 epochs). 1e-5 is optimal.
        - Only image encoder parameters with requires_grad=True are passed to
          Adam — the frozen text encoder parameters are excluded.
        - Best checkpoint is saved by validation accuracy (not val loss) since
          accuracy is the primary evaluation metric.
        - Training history is saved as JSON for plotting learning curves.

    Args:
        use_short_labels: If True, use short label strings as text anchors
                          instead of paragraphs (Ablation A).
        run_name:         Used as the filename prefix for saved weights and
                          history, e.g. "clip_base" → "clip_base_best.pth".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CLIP uses its own normalization stats, not ImageNet
    train_loader, val_loader, _ = make_loaders(data_dir, batch_size=batch_size, norm="clip")

    model, _ = load_clip(device, clip_model_name)
    model     = freeze_text_encoder(model)  # freeze text side before encoding

    # Pre-encode text anchors once — these are fixed for the entire training run.
    # torch.no_grad() is redundant here since the text encoder is frozen,
    # but makes the intent explicit.
    with torch.no_grad():
        if use_short_labels:
            text_features = encode_short_labels(model, device)
            print("Using SHORT label text anchors (Ablation A)")
        else:
            text_features = encode_descriptions(model, descriptions_path, device)
            print("Using RICH PARAGRAPH text anchors")

    # Only optimize parameters that were not frozen by freeze_text_encoder()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )

    stopper     = EarlyStopping(patience=patience)
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    best_path   = os.path.join(save_dir, f"{run_name}_best.pth")

    start = time.time()
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, text_features,
                                  optimizer, device, train=True)
        v_loss, v_acc = run_epoch(model, val_loader,   text_features,
                                  optimizer, device, train=False)

        # Record history for training curve plots
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:3d} | train loss {t_loss:.4f} acc {t_acc:.4f} "
              f"| val loss {v_loss:.4f} acc {v_acc:.4f}")

        # Save checkpoint whenever validation accuracy improves
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  → Saved best model (val acc {best_val_acc:.4f})")

        # Check early stopping criterion on validation loss
        if stopper.step(v_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min. Best val acc: {best_val_acc:.4f}")

    # Save full history for notebook training curve plots
    history_path = os.path.join(save_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    return history, best_val_acc


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         default="data/raw")
    parser.add_argument("--descriptions",     default="data/descriptions/landmarks.json")
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--patience",         type=int,   default=5)
    parser.add_argument("--save_dir",         default="models")
    parser.add_argument("--run_name",         default="clip_run")
    parser.add_argument("--use_short_labels", action="store_true",
                        help="Use short label strings instead of paragraphs (Ablation A)")
    parser.add_argument("--clip_model",       default="ViT-B/32")
    args = parser.parse_args()

    train(
        data_dir          = args.data_dir,
        descriptions_path = args.descriptions,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        patience          = args.patience,
        save_dir          = args.save_dir,
        run_name          = args.run_name,
        use_short_labels  = args.use_short_labels,
        clip_model_name   = args.clip_model,
    )
