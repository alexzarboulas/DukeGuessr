# DukeGuessr

**Author:** Alexander Zarboulas

**Identify Duke University landmarks from uploaded photos using CLIP.**

Upload a photo taken on Duke's campus and DukeGuessr will match it against rich text descriptions of 7 landmark classes using CLIP's shared image-text embedding space, displaying the predicted location and confidence scores.

---

## Real-World Motivation

- **Campus navigation:** new students and visitors can identify where a photo was taken
- **Historical photo archiving:** locate undated archive photos by visual content alone
- **Social media geotagging:** automatically suggest the correct Duke location tag

This project builds on CLIP (Radford et al., 2021) and connects to the visual place recognition (VPR) research area, including systems like NetVLAD and GeoGuessr-style localization.

---

## What It Does

DukeGuessr is a web application that identifies Duke University landmarks from uploaded photos using three machine learning models running in parallel: a fine-tuned CLIP model (the primary model), a zero-shot CLIP baseline, and a fine-tuned ViT-B/16 for architecture comparison. Instead of numeric class labels, the CLIP-based models use natural language paragraphs as class anchors — CLIP learns to match what it *sees* in a photo with what it *reads* about a landmark, and the best-matching description determines the prediction. For every upload, the app also generates an attention brightness map showing which parts of the image most influenced the fine-tuned CLIP model's decision.

**7 classes:**
Perkins Library · Main Quad · Duke Chapel · Campus Dr / Bus Stop · Sarah P. Duke Gardens · Wannamaker Benches · Other

**Three architectures compared side-by-side in the app:**

| Model | Architecture | Fine-Tuned | Training Objective | Text Anchors | Test Accuracy |
|-------|-------------|:----------:|-------------------|:------------:|:-------------:|
| Zero-Shot CLIP | ViT-B/32 | No | None (pretrained weights only) | 7 landmark paragraphs | 32.1% |
| Fine-Tuned CLIP | ViT-B/32 | Yes (image encoder) | Cosine similarity vs. frozen text anchors, cross-entropy | 7 landmark paragraphs | 100.0% |
| Fine-Tuned ViT-B/16 | ViT-B/16 | Yes (full model) | CrossEntropyLoss over integer class labels | None | 100.0% |

### Attention Brightness Map

For every uploaded photo, the app generates an **attention brightness map** using the fine-tuned CLIP model's internal attention weights. A forward hook is registered on the last ViT-B/32 transformer block during inference. The hook recomputes softmax(QK^T / sqrt(d)) across all 12 attention heads, averages them, and extracts the CLS token's attention over the 49 image patches (7x7 grid). Those weights are upscaled to the original image resolution via bilinear interpolation and used to modulate pixel brightness: high-attention regions appear at full brightness, low-attention regions dim toward black.

This reveals *where* the model looked when making its prediction, at zero extra inference cost (no second forward pass required).

---

## Quick Start

```bash
git clone https://github.com/alexzarboulas/DukeGuessr.git
cd DukeGuessr
bash setup.sh   # one-time: installs dependencies and downloads model weights
bash start.sh   # starts backend and frontend in one terminal
```

Open the URL printed in the terminal (typically http://localhost:3000).

See **[SETUP.md](SETUP.md)** for full installation details and troubleshooting.

---

## Evaluation

### Accuracy

| Model | Test Accuracy | Macro F1 | Train Time |
|-------|:------------:|:--------:|:----------:|
| Random baseline | 14.3% | 0.14 | — |
| Zero-Shot CLIP (short labels) | 49.1% | — | 0 |
| Zero-Shot CLIP (rich paragraphs) | 32.1% | — | 0 |
| Fine-Tuned CLIP (lr=1e-5) | **100.0%** | **1.00** | 16.7 min |
| Fine-Tuned ViT-B/16 (lr=1e-4) | **100.0%** | **1.00** | 18.3 min |

Evaluated on 53 strictly held-out test images (15% stratified split, seed=42, never seen during training). Zero failures on the test set for both fine-tuned models.

### Learning Rate Sweep (Fine-Tuned CLIP)

| Learning Rate | Best Val Accuracy |
|:---:|:---:|
| 1e-4 | 39.6% |
| **1e-5** | **100.0%** |
| 1e-6 | 96.2% |

`lr=1e-5` was selected. `lr=1e-4` overshot and degraded the pretrained image encoder weights; `lr=1e-6` converged too slowly to reach peak accuracy within the 20-epoch budget.

### Ablations

**Ablation A: Text anchor quality (zero-shot CLIP):**
Short class-name strings (`"Duke Chapel"`) outperformed 200-word visual paragraphs (49.1% vs 32.1%). This is a consequence of CLIP's 77-token input limit: the paragraphs are ~200 words and are silently truncated, cutting off most of the visual description. Visual features were front-loaded in each paragraph to mitigate this, but short labels proved more robust zero-shot.

**Ablation B: Augmentation impact:**
Re-training without the four augmentation techniques (random crop, horizontal flip, rotation, color jitter) produced no accuracy drop on the test set, likely because the dataset is small enough that the fine-tuned model memorizes it either way. Augmentation is retained as a regularization safeguard.

Per-class precision/recall/F1 and training curves: `notebooks/experiments.ipynb`.

---

## Conclusions

1. **Fine-tuning is essential for CLIP on a domain-specific task.** Zero-shot CLIP peaks at 49.1% on a 7-class problem a random guesser solves at 14.3%. Conservative fine-tuning at `lr=1e-5` closes that gap entirely, from 32.1% to 100%.

2. **Learning rate is the most critical hyperparameter for CLIP fine-tuning.** A 10× increase (`lr=1e-4`) destroyed the pretrained image encoder features and dropped accuracy to 39.6%, worse than zero-shot. Transfer learning at this scale requires a conservative rate to preserve pretrained representations while adapting them.

3. **Short text anchors outperform rich descriptions in zero-shot use.** The 77-token CLIP limit is a real architectural constraint. Concise labels are more reliable than long paragraphs that get truncated. Once fine-tuned, this distinction disappears.

4. **350 images is sufficient for a well-constrained classification task.** With a pretrained backbone and appropriate fine-tuning, 245 training images (70%) was enough for perfect generalization across 7 classes. The bottleneck is backbone quality, not dataset size.

---

## Next Steps

- **More landmarks:** the current 7 classes cover only a fraction of Duke's campus. Expanding to 20-30 landmarks with ~50 images each would stress-test both models.
- **Harder negatives in "Other":** the current "other" class is generic campus scenes. Adding near-miss images (similar lighting, similar architecture from off-campus) would test the confidence threshold more rigorously.
- **Public deployment:** hosting the backend on a GPU endpoint (e.g., Modal, Replicate) and the frontend on Vercel would make the demo accessible without local setup.
- **Seasonal robustness:** all photos were taken in April 2026. Testing on photos from other seasons (fall foliage, winter snow) would reveal how much the model depends on color and lighting conditions specific to spring.

---

## Video Links

- **Demo video (3–5 min):** [link TBD]
- **Technical walkthrough (5–10 min):** [link TBD]
