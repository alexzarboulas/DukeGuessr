"""
vit_model.py — Vision Transformer (ViT-B/16) classifier for DukeGuessr.

What this file does:
    Defines the ViT-B/16 comparison model used alongside fine-tuned CLIP.
    Takes a pretrained ImageNet ViT, replaces its 1000-class head with a
    7-class linear layer, and provides a checkpoint loader for inference.

Academic contributions demonstrated here:
    - Fine-tuning a pretrained vision transformer (ViT-B/16) on a custom
      7-class image classification task. The backbone is initialized from
      ImageNet-1K weights and the classification head is replaced.
    - Architecture comparison: this model represents the traditional
      closed-set classification paradigm (image → integer label via
      CrossEntropyLoss), directly contrasted with CLIP's contrastive
      image-text matching approach. Both models achieve 100% test accuracy,
      but with fundamentally different learning mechanisms.
    - The key architectural difference from CLIP: ViT maps images to integer
      class indices with no semantic grounding. Adding a new landmark class
      requires retraining; there are no language anchors that generalize
      beyond the training distribution.

AI attribution: This file was scaffolded with Claude (Anthropic). Architecture
selection (ViT-B/16 as the comparison model) and the decision to use ImageNet
normalization are Alexander Zarboulas's original work. See ATTRIBUTION.md for full details.
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 7


def build_vit(freeze_backbone: bool = False) -> nn.Module:
    """
    Construct a ViT-B/16 model adapted for 7-class Duke landmark classification.

    Architecture changes from the stock ImageNet model:
        - model.heads.head replaced with nn.Linear(768, 7)
        - All other weights kept from ImageNet pretraining (transfer learning)

    The classification head input dimension is 768 — the hidden dimension of
    ViT-B/16's transformer encoder output (patch_size=16, 12 layers, 12 heads).

    Args:
        freeze_backbone: If True, freeze all layers except the new head.
                         Used in ablation experiments to isolate the effect
                         of fine-tuning the full backbone vs. head-only.
                         False (default) fine-tunes the entire network.

    Returns:
        nn.Module ready for training or checkpoint loading.
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze everything first, then unfreeze the new head below
        for param in model.parameters():
            param.requires_grad = False

    # Replace the 1000-class ImageNet head with a 7-class head.
    # in_features = 768 for ViT-B/16.
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    # The new linear layer has requires_grad=True by default,
    # so it trains even when freeze_backbone=True.

    return model


def load_checkpoint(path: str, model: nn.Module, device: torch.device) -> nn.Module:
    """
    Load a saved state dict into a model and move it to the target device.

    Args:
        path:   path to the .pth file saved by train_vit.py
        model:  an uninitialized build_vit() instance
        device: cpu or cuda

    Returns:
        model with loaded weights, moved to device, ready for inference.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model
