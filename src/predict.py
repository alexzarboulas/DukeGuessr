"""
predict.py — Single-image inference for DukeGuessr (CLIP and ViT).

What this file does:
    Provides the inference functions called by the Flask backend for every
    uploaded image. Handles image decoding, preprocessing, model forward
    pass, softmax probability conversion, confidence thresholding, and
    result formatting for all three model variants served by the app.

    For fine-tuned CLIP, also generates a brightness map: the original image
    with pixel values scaled by the model's per-patch attention weights.
    High-attention regions appear at full brightness; low-attention regions
    are dimmed toward black. This is captured at zero extra inference cost
    by registering a forward hook on the last ViT transformer block that
    fires during the existing encode_image() call.

Academic contributions demonstrated here:
    - Deployment-ready inference pipeline connecting trained models to the
      web application. Both CLIP (cosine similarity) and ViT (linear logits)
      are handled through a unified result format.
    - Correct normalization at inference time: CLIP predictions use CLIP's
      pretraining mean/std; ViT predictions use ImageNet mean/std. Using
      the wrong stats would degrade accuracy without any training-time warning.
    - Confidence thresholding: predictions below 30% confidence are treated
      as "other" regardless of the top class, providing a safety net against
      overconfident wrong predictions on out-of-distribution images.
    - Attention visualization via forward hook: captures Q and K from the
      last ViT-B/32 block during the existing inference pass, computes
      softmax(QK^T / sqrt(d)) per head, averages across heads, and maps
      CLS-to-patch attention weights back to image space.
    - The shared _build_result() helper ensures consistent JSON output
      format across all three models, making it straightforward to display
      and compare them side-by-side in the frontend.

AI attribution: This file was scaffolded with Claude (Anthropic). The confidence
threshold choice (0.30), the decision to expose all three models through a unified
result format, normalization stat choices, and the attention visualization design
are Alexander Zarboulas's original work. See ATTRIBUTION.md for full details.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data import CLASS_NAMES

# ── Normalization constants ───────────────────────────────────────────────────
# Must match the normalization used during training for each model type.
# Mixing these up (e.g. using ImageNet stats for CLIP inference) would silently
# hurt accuracy with no error message.
CLIP_MEAN     = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD      = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# GPS coordinates and display names for each landmark class.
# "other" coordinates are the approximate campus centroid — shown only in
# edge cases where the confidence threshold forces a fallback.
LANDMARK_INFO = {
    "perkins":            {"name": "Perkins Library",       "lat": 35.9996, "lon": -78.9408},
    "main_quad":          {"name": "Main Quad",             "lat": 36.0015, "lon": -78.9410},
    "chapel":             {"name": "Duke Chapel",           "lat": 36.0017, "lon": -78.9422},
    "bus_stop":           {"name": "Campus Dr / Bus Stop",  "lat": 36.0010, "lon": -78.9400},
    "gardens":            {"name": "Sarah P. Duke Gardens", "lat": 36.0023, "lon": -78.9404},
    "wannamaker_benches": {"name": "Wannamaker Benches",    "lat": 36.0003, "lon": -78.9268},
    "other":              {"name": "Unknown Location",      "lat": 36.0012, "lon": -78.9400},
}

# Minimum softmax probability for a named landmark prediction to be shown.
# Below this threshold the prediction is treated as "other" regardless of
# the top class — prevents overconfident wrong predictions on ambiguous photos.
CONFIDENCE_THRESHOLD = 0.30

# ── Preprocessing pipelines ───────────────────────────────────────────────────
# Defined once at module load and reused for every request — avoids
# reconstructing the transform pipeline on every inference call.
_preprocess_clip = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

_preprocess_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Shared result builder ─────────────────────────────────────────────────────
def _build_result(probs: list, top_idx: int, top_class: str) -> dict:
    """
    Convert raw softmax probabilities into the JSON response dict.

    Applies confidence thresholding: if the top predicted class is "other"
    or its probability is below CONFIDENCE_THRESHOLD, is_other is set True
    and the frontend shows the "unknown location" state instead of a landmark.

    All 7 class probabilities are returned sorted by confidence so the
    frontend can render the full comparison bar chart.

    Returns:
        {
          "landmark":        display name of the predicted class,
          "confidence":      softmax probability of the top class (0–1),
          "latitude":        GPS latitude of the predicted landmark,
          "longitude":       GPS longitude,
          "is_other":        True if prediction is low-confidence or "other",
          "all_predictions": list of {landmark, confidence} dicts, sorted desc
        }
    """
    top_conf      = probs[top_idx]
    is_other      = (top_class == "other") or (top_conf < CONFIDENCE_THRESHOLD)
    display_class = "other" if is_other else top_class
    info          = LANDMARK_INFO.get(display_class, LANDMARK_INFO["other"])

    all_preds = sorted(
        [{"landmark":   LANDMARK_INFO.get(CLASS_NAMES[i], {}).get("name", CLASS_NAMES[i]),
          "confidence": round(probs[i], 4)}
         for i in range(len(CLASS_NAMES))],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "landmark":        info["name"],
        "confidence":      round(top_conf, 4),
        "latitude":        info["lat"],
        "longitude":       info["lon"],
        "is_other":        is_other,
        "all_predictions": all_preds,
    }


# ── Attention brightness map ──────────────────────────────────────────────────
def _make_brightness_map(attn_store: list, image_pil: Image.Image) -> str | None:
    """
    Convert captured attention weights into a brightness map PNG (base64).

    Mechanism:
        CLIP ViT-B/32 processes the 224×224 image as a 7×7 grid of 32×32
        patches (49 patches + 1 CLS token = 50 sequence positions). The last
        transformer block's multi-head attention produces a (heads, 50, 50)
        weight matrix. Row 0 of that matrix is the CLS token attending to
        every other token — columns 1–49 are how much the model "looked at"
        each image patch when forming its global representation.

        We average CLS-to-patch attention across all 12 heads, apply a mild
        gamma (0.6) to prevent the map from being too dark in mid-attention
        regions, upscale from 7×7 to the original image resolution via
        bilinear interpolation, and multiply the original RGB pixel values
        by the normalized [0, 1] mask. The result: high-attention regions
        appear at full brightness, low-attention regions dim toward black.

    Args:
        attn_store: list populated by the forward hook (length 1 after inference)
        image_pil:  original PIL image before any CLIP preprocessing

    Returns:
        base64-encoded PNG string, or None if the hook produced no data.
    """
    if not attn_store:
        return None

    # attn_store[0]: (batch=1, num_heads=12, seq_len=50, seq_len=50)
    attn       = attn_store[0][0]              # (12, 50, 50)
    patch_attn = attn.mean(dim=0)[0, 1:].numpy()  # (49,) — CLS → each patch

    # Normalize to [0, 1] then apply gamma to lift mid-tone visibility
    patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)
    patch_attn = patch_attn ** 0.6  # gamma < 1 brightens mid-attention areas

    # ViT-B/32: 7×7 patch grid — derive from patch count to stay architecture-agnostic
    grid_size = int(round(patch_attn.shape[0] ** 0.5))     # 7
    attn_grid = patch_attn.reshape(grid_size, grid_size)   # (7, 7)

    # Upscale attention grid to original image dimensions
    orig_w, orig_h = image_pil.size
    attn_pil = Image.fromarray((attn_grid * 255).astype(np.uint8), mode="L")
    attn_pil = attn_pil.resize((orig_w, orig_h), Image.BILINEAR)
    mask     = np.array(attn_pil).astype(np.float32) / 255.0   # (H, W)

    # Scale original RGB pixels by attention mask
    orig_np    = np.array(image_pil).astype(np.float32)         # (H, W, 3)
    brightness = np.clip(orig_np * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Encode to base64 PNG for JSON transport
    out_img = Image.fromarray(brightness, mode="RGB")
    buf     = io.BytesIO()
    out_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── CLIP inference ────────────────────────────────────────────────────────────
def predict(model, text_features: torch.Tensor, image_bytes: bytes,
            device: torch.device) -> dict:
    """
    Run CLIP inference and generate an attention brightness map.

    Classification: image embedding vs. 7 pre-encoded text anchor embeddings
    via cosine similarity (scaled by logit_scale) → softmax → argmax.

    Attention visualization: a forward hook is registered on the last ViT
    transformer block before the forward pass. The hook manually computes
    softmax(QK^T / sqrt(d)) for each head using the module's in_proj_weight
    and in_proj_bias — this is equivalent to what the block computes
    internally but gives us access to the weights that are otherwise
    discarded when need_weights=False. The hook fires once during the
    existing compute_logits() call, adding negligible overhead.

    Args:
        model:         CLIP model (fine-tuned or base)
        text_features: (7, 512) pre-encoded, normalized text embeddings
        image_bytes:   raw bytes from the uploaded file
        device:        cpu or cuda

    Returns:
        Result dict from _build_result() plus "brightness_map" (base64 PNG).
    """
    from clip_model import compute_logits

    # Decode raw bytes → PIL image (keep original for brightness map)
    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess_clip(image).unsqueeze(0).to(device)

    # ── Register attention hook ───────────────────────────────────────────────
    # Hooks into the last ViT block's nn.MultiheadAttention. Because CLIP calls
    # attn(..., need_weights=False), the weights are not returned normally.
    # We recompute them from Q and K inside the hook at zero extra cost.
    attn_store  = []
    last_block  = model.visual.transformer.resblocks[-1]

    def _attn_hook(module, inp, out):
        x                        = inp[0]                   # (seq_len, batch, embed_dim)
        seq_len, batch, embed_dim = x.shape
        num_heads                = module.num_heads
        head_dim                 = embed_dim // num_heads

        # Recompute Q and K from the combined in_proj matrix
        qkv     = F.linear(x, module.in_proj_weight, module.in_proj_bias)
        q, k, _ = qkv.chunk(3, dim=-1)

        # Reshape to (batch*heads, seq_len, head_dim) for batched matmul
        q = q.reshape(seq_len, batch * num_heads, head_dim).transpose(0, 1)
        k = k.reshape(seq_len, batch * num_heads, head_dim).transpose(0, 1)

        scale = head_dim ** -0.5
        attn  = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        # Store as (batch, heads, seq_len, seq_len) on CPU
        attn_store.append(attn.reshape(batch, num_heads, seq_len, seq_len).detach().cpu())

    handle = last_block.attn.register_forward_hook(_attn_hook)

    model.eval()
    with torch.no_grad():
        # Hook fires here during model.encode_image() inside compute_logits
        logits = compute_logits(model, tensor, text_features)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    handle.remove()   # always remove — don't leave hooks registered

    top_idx   = int(torch.tensor(probs).argmax())
    top_class = CLASS_NAMES[top_idx]
    result    = _build_result(probs, top_idx, top_class)

    # Attach brightness map — generated from attention captured during inference
    result["brightness_map"] = _make_brightness_map(attn_store, image)
    return result


# ── ViT inference ─────────────────────────────────────────────────────────────
def predict_vit(model, image_bytes: bytes, device: torch.device) -> dict:
    """
    Run ViT-B/16 inference on a raw image uploaded via the web app.

    Classification: image → linear head logits (7 values) → softmax → argmax.
    No text encoding or cosine similarity — purely image-to-label mapping.

    Uses ImageNet normalization (not CLIP stats) to match the ViT's
    pretraining distribution.

    Args:
        model:       fine-tuned ViT-B/16 with 7-class head
        image_bytes: raw bytes from the uploaded file
        device:      cpu or cuda

    Returns:
        Result dict from _build_result() — same format as CLIP predict()
        so the frontend can display all three models identically.
        No brightness_map field (attention architecture differs from CLIP ViT).
    """
    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess_imagenet(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    top_idx   = int(torch.tensor(probs).argmax())
    top_class = CLASS_NAMES[top_idx]
    return _build_result(probs, top_idx, top_class)
