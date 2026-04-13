"""
evaluate.py — Model evaluation, metrics, confusion matrix, and error analysis.

What this file does:
    Runs a trained model (CLIP or ViT) against the held-out test set and
    produces the full suite of evaluation outputs: overall accuracy, per-class
    precision/recall/F1, a confusion matrix heatmap, and a visualization of
    misclassified images. Supports both CLIP and ViT architectures through a
    shared interface.

Academic contributions demonstrated here:
    - Multiple evaluation metrics beyond accuracy: per-class precision,
      recall, and F1-score via sklearn's classification_report, plus macro
      F1 to measure balanced performance across all 7 classes.
    - Error analysis with visualization: the confusion matrix heatmap
      (seaborn) identifies which class pairs the model confuses most often.
      The failure case grid shows misclassified images alongside their true
      and predicted labels, enabling qualitative analysis of why the model
      fails on specific examples.
    - Controlled evaluation: the test set was held out via stratified split
      before any training occurred and is never used for model selection.
      All reported numbers reflect true generalization performance.
    - Architecture comparison support: the same evaluation pipeline runs for
      both CLIP (contrastive) and ViT (classification) models, enabling a
      fair apples-to-apples comparison of their error distributions.

AI attribution: This file was scaffolded with Claude (Anthropic). Analysis
decisions (evaluation metric selection, conclusions drawn from results) are
Alexander Zarboulas's original work. See ATTRIBUTION.md for full details.

Usage:
    # Evaluate fine-tuned CLIP
    python src/evaluate.py --model clip \
        --weights models/clip_base_best.pth \
        --descriptions data/descriptions/landmarks.json \
        --data_dir data/raw

    # Evaluate fine-tuned ViT
    python src/evaluate.py --model vit \
        --weights models/vit_base_best.pth \
        --data_dir data/raw
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data import make_splits, make_transforms, DukeLandmarkDataset, CLASS_NAMES
from torch.utils.data import DataLoader


# ── Prediction collection helpers ─────────────────────────────────────────────
def _get_clip_preds(model, text_features, loader, device):
    """
    Run CLIP inference over a full DataLoader and collect predictions.

    Classification is via argmax over cosine similarity logits —
    same mechanism used during training and inference.

    Returns:
        labels: (N,) ground-truth class indices
        preds:  (N,) predicted class indices
    """
    from clip_model import compute_logits
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = compute_logits(model, images, text_features)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)


def _get_vit_preds(model, loader, device):
    """
    Run ViT inference over a full DataLoader and collect predictions.

    Classification is via argmax over the 7-class linear head logits.

    Returns:
        labels: (N,) ground-truth class indices
        preds:  (N,) predicted class indices
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            all_preds.extend(model(images).argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)


# ── Confusion matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, class_names,
                          save_path="docs/confusion_matrix.png"):
    """
    Generate and save a confusion matrix heatmap.

    The confusion matrix (C[i,j] = number of samples with true class i
    predicted as class j) makes it immediately clear which classes are
    confused with each other. Diagonal = correct predictions.

    Uses seaborn's heatmap with annotation (fmt="d" for integer counts)
    and a blue colormap to match the Duke theme.
    """
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


# ── Failure case visualization ─────────────────────────────────────────────────
def show_failures(model, test_paths, test_labels, class_names, device,
                  model_type="clip", text_features=None, n=10,
                  save_path="docs/failure_cases.png"):
    """
    Find and visualize misclassified test images.

    Iterates through the test set image-by-image, runs inference, and
    collects up to n cases where the prediction differs from the ground
    truth label. Displays them in a grid with true vs. predicted labels.

    This qualitative analysis reveals patterns the confusion matrix cannot:
    - Which visual features caused the confusion (e.g., similar architecture)
    - Whether failures cluster around specific class pairs
    - Whether misclassified images are genuinely ambiguous or clear errors

    Args:
        model_type:    "clip" or "vit" — determines preprocessing normalization
                       and whether to use cosine similarity or linear logits
        text_features: required when model_type="clip"; ignored for "vit"
        n:             maximum number of failure cases to show
    """
    from clip_model import compute_logits

    # Use the same non-augmented transform as test evaluation
    norm      = "clip" if model_type == "clip" else "imagenet"
    transform = make_transforms(norm=norm, augment=False)

    model.eval()
    failures = []

    for path, label in zip(test_paths, test_labels):
        from PIL import Image
        img    = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "clip":
                logits = compute_logits(model, tensor, text_features)
            else:
                logits = model(tensor)

        pred = logits.argmax(1).item()
        if pred != label:
            failures.append((img, label, pred))
        if len(failures) >= n:
            break

    if not failures:
        print("No failures found — perfect test accuracy.")
        return

    cols = min(5, len(failures))
    rows = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, (img, true_lbl, pred_lbl) in enumerate(failures):
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]}",
            fontsize=8
        )
        axes[i].axis("off")

    # Hide any unused subplot slots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Failure Cases", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


# ── Main evaluation function ───────────────────────────────────────────────────
def evaluate(data_dir, weights_path, model_type="clip",
             descriptions_path="data/descriptions/landmarks.json", batch_size=32):
    """
    Full evaluation pipeline for one model on the held-out test set.

    Steps:
        1. Reconstruct the same stratified split used during training
           (same random seed=42) to recover the exact test set
        2. Run inference over all test images
        3. Print accuracy and per-class precision/recall/F1
        4. Save confusion matrix heatmap to docs/
        5. Save failure case grid to docs/

    The test set (53 images, 15% of 350 total) was strictly held out —
    never used for training or hyperparameter selection.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct the same split using the fixed seed — guarantees we evaluate
    # on exactly the same images that were held out during training
    _, _, _, _, test_paths, test_labels = make_splits(data_dir)

    norm        = "clip" if model_type == "clip" else "imagenet"
    test_set    = DukeLandmarkDataset(test_paths, test_labels,
                                      make_transforms(norm=norm, augment=False))
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    if model_type == "clip":
        from clip_model import load_finetuned_clip
        model, text_features = load_finetuned_clip(weights_path, device,
                                                    descriptions_path)
        labels, preds = _get_clip_preds(model, text_features,
                                        test_loader, device)
    else:
        from vit_model import build_vit, load_checkpoint
        model         = load_checkpoint(weights_path, build_vit(), device)
        text_features = None
        labels, preds = _get_vit_preds(model, test_loader, device)

    # Overall accuracy
    acc = (labels == preds).mean()
    print(f"\nTest Accuracy: {acc*100:.2f}%\n")

    # Per-class precision, recall, F1 + macro averages
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # Confusion matrix heatmap
    plot_confusion_matrix(labels, preds, CLASS_NAMES)

    # Failure case visualization
    failure_path = f"docs/failure_cases_{model_type}.png"
    show_failures(model, test_paths, test_labels, CLASS_NAMES, device,
                  model_type=model_type, text_features=text_features,
                  save_path=failure_path)

    return acc, labels, preds


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="data/raw")
    parser.add_argument("--weights",      required=True)
    parser.add_argument("--model",        default="clip", choices=["clip", "vit"])
    parser.add_argument("--descriptions", default="data/descriptions/landmarks.json")
    parser.add_argument("--batch_size",   type=int, default=32)
    args = parser.parse_args()

    evaluate(args.data_dir, args.weights, args.model,
             args.descriptions, args.batch_size)
