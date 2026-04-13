"""
clip_model.py — CLIP model loading, text encoding, and fine-tuning setup.

What this file does:
    Provides all CLIP-specific operations used by the training and inference
    pipelines: loading the pretrained model, freezing the text encoder,
    encoding landmark descriptions into fixed text embeddings, and computing
    image-text cosine similarity logits for classification.

Academic contributions demonstrated here:
    - Adaptation of a vision-language model (OpenAI CLIP ViT-B/32) for a
      custom classification task. Rather than adding a traditional linear
      head, the model is repurposed to perform contrastive matching between
      image embeddings and natural language class descriptions.
    - Transfer learning design: the text encoder is frozen throughout
      training so its pretrained language representations remain intact.
      Only the image encoder is updated, aligning it toward the fixed
      semantic anchors.
    - Feature engineering via text embeddings: the 7 landmark paragraph
      descriptions are encoded once into 512-dimensional vectors that serve
      as class prototypes in the shared embedding space. Classification is
      then cosine similarity rather than a learned weight matrix.
    - Modular separation of concerns — this file owns CLIP-specific logic;
      training loops, evaluation, and inference import from here rather than
      reimplementing it.

Token limit note:
    openai/clip truncates text at 77 tokens (~50 words). The landmark
    paragraphs in landmarks.json are ~200 words, so the tail of each is
    dropped. Visual information is front-loaded in each description to
    minimize the impact of truncation. truncate=True lets this fail
    gracefully rather than raising an error.

AI attribution: This file was scaffolded with Claude (Anthropic). Decisions
about which layers to freeze, the 77-token truncation workaround strategy, and
the choice to freeze the text encoder are Alexander Zarboulas's original work. See
ATTRIBUTION.md for full details.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    print("WARNING: 'clip' package not found. Run: pip install git+https://github.com/openai/CLIP.git")

from data import CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)  # 7


def load_clip(device: torch.device, model_name: str = "ViT-B/32"):
    """
    Load the pretrained CLIP model and its image preprocessor.

    model.float() converts from float16 (CLIP's default) to float32,
    which is required for stable gradient computation during fine-tuning.
    Float16 can cause gradient underflow with small learning rates like 1e-5.

    Returns:
        model      : CLIP model in float32
        preprocess : torchvision transform pipeline for CLIP's image encoder
    """
    assert _CLIP_AVAILABLE, "clip package required"
    model, preprocess = clip.load(model_name, device=device)
    model = model.float()  # float16 → float32 for fine-tuning stability
    return model, preprocess


def freeze_text_encoder(model):
    """
    Freeze all parameters in CLIP's text encoder branch.

    Frozen components:
        - transformer (the 12-layer text transformer)
        - token_embedding (vocabulary embedding table)
        - ln_final (final layer norm on text side)
        - positional_embedding (text position encodings)
        - text_projection (projects text transformer output → 512-d space)

    Why freeze: The paragraph descriptions are fixed class anchors. There is
    no benefit to updating the text encoder since it will always see the same
    7 inputs. Freezing it also reduces the number of trainable parameters,
    which acts as an implicit regularizer and speeds up training.

    The image encoder (visual transformer) and logit_scale remain trainable.

    Returns:
        model with text encoder parameters frozen in-place
    """
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.ln_final.parameters():
        param.requires_grad = False
    model.positional_embedding.requires_grad = False
    model.text_projection.requires_grad = False
    return model


def encode_descriptions(model, descriptions_path: str, device: torch.device) -> torch.Tensor:
    """
    Tokenize and encode all 7 landmark paragraph descriptions into text embeddings.

    This is called once at startup (or once before training) and the result is
    cached — the text encoder is never called again during training since the
    descriptions are fixed.

    The output is L2-normalized so cosine similarity reduces to a dot product,
    which is what CLIP's logit_scale multiplication expects.

    Args:
        model:             CLIP model (used only for its text encoder here)
        descriptions_path: path to landmarks.json
        device:            target device for the output tensor

    Returns:
        text_features: (7, 512) float32 tensor of normalized text embeddings,
                       one row per class in CLASS_NAMES order.
    """
    assert _CLIP_AVAILABLE
    with open(descriptions_path) as f:
        descriptions = json.load(f)

    # Maintain CLASS_NAMES order so index i in text_features matches label i
    paragraphs = [descriptions[cls] for cls in CLASS_NAMES]

    # truncate=True silently drops tokens beyond position 77 rather than raising
    tokens = clip.tokenize(paragraphs, truncate=True).to(device)

    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        # L2-normalize each embedding vector to unit length
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # (7, 512), normalized


def encode_short_labels(model, device: torch.device) -> torch.Tensor:
    """
    Encode short human-readable class label strings instead of paragraphs.

    Used in Ablation A to measure the impact of rich paragraph descriptions
    vs. short labels. Short labels fit well within the 77-token limit and
    match the style of text CLIP was pretrained on (short internet captions),
    which is why they outperform paragraphs in zero-shot evaluation (49.1%
    vs 32.1%).

    Returns:
        text_features: (7, 512) normalized text embeddings, same format as
                       encode_descriptions.
    """
    assert _CLIP_AVAILABLE
    labels = [
        "Perkins Library at Duke University",
        "Main Quad at Duke University",
        "Duke Chapel",
        "Campus Drive bus stop at Duke University",
        "Sarah P. Duke Gardens",
        "Wannamaker Benches outdoor seating East Campus Duke University",
        "Other Duke University campus location",
    ]
    tokens = clip.tokenize(labels, truncate=True).to(device)
    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # (7, 512), normalized


def compute_logits(model, images: torch.Tensor,
                   text_features: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images and compute classification logits via cosine similarity.

    The logit for class i is:
        logit_i = exp(logit_scale) * cos_sim(image_embedding, text_feature_i)

    logit_scale is CLIP's learned temperature parameter (initialized to log(1/0.07)).
    Scaling cosine similarities before softmax sharpens the distribution,
    making the model more confident on clear matches.

    Args:
        images:        (B, 3, 224, 224) preprocessed image batch
        text_features: (7, 512) pre-encoded, L2-normalized text embeddings

    Returns:
        logits: (B, 7) scaled cosine similarities — pass to F.cross_entropy
    """
    # Encode images and L2-normalize to unit vectors
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Scaled dot product = scaled cosine similarity (both sides are unit vectors)
    logits = model.logit_scale.exp() * (image_features @ text_features.T)
    return logits


def load_finetuned_clip(weights_path: str, device: torch.device,
                        descriptions_path: str, model_name: str = "ViT-B/32"):
    """
    Load a fine-tuned CLIP checkpoint and pre-encode text features.

    Used at inference time (app.py) to initialize the model once at startup.
    Text features are encoded here and reused for every prediction request.

    Args:
        weights_path:      path to .pth file containing the fine-tuned state dict
        device:            target device
        descriptions_path: path to landmarks.json
        model_name:        must match the architecture used during training

    Returns:
        (model, text_features) — model in eval mode, text_features (7, 512)
    """
    model, _ = load_clip(device, model_name)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # Pre-encode text anchors once — reused for every inference call
    text_features = encode_descriptions(model, descriptions_path, device)
    return model, text_features
