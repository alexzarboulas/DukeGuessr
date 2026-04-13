# Attribution

## Dataset

All 350 landmark photos were **collected originally** by Alexander Zarboulas using a personal camera during dedicated on-campus photography sessions at Duke University in April 2026. No third-party images were used. Photos are not redistributed in this repository.

**7 classes collected:** Perkins Library, Main Quad, Duke Chapel, Campus Dr / Bus Stop, Sarah P. Duke Gardens, Wannamaker Benches, Other. ~50 images per class.

---

## AI Development Tools

AI tools (Claude Code, Anthropic) were used throughout this project for code scaffolding, debugging, and frontend design. Here is a substantive account of what was generated, what was modified, and what required manual work:

| Area | What AI generated | What I wrote / debugged / decided |
|------|-------------------|-----------------------------------|
| `src/data.py` | Initial dataset class, transform pipeline, split logic | Normalization stat choices (CLIP vs ImageNet), augmentation technique selection, stratification parameters |
| `src/clip_model.py` | CLIP loading wrappers, text encoding, logit computation | Decision to freeze text encoder, which layers to freeze, 77-token truncation workaround strategy |
| `src/train_clip.py` | Training loop structure, early stopping class | All hyperparameter choices (lr=1e-5, patience=5, weight_decay=1e-4), experiment design decisions |
| `src/train_vit.py` | ViT training loop | Hyperparameter choices (lr=1e-4), decision to use ImageNet normalization |
| `src/evaluate.py` | Evaluation pipeline structure, per-class metrics output | Analysis write-up, evaluation metric selection, conclusions drawn from results |
| `src/predict.py` | Single-image inference, GPS lookup, brightness map generation | Confidence threshold choice (0.30), decision to expose all 3 models in one response, attention visualization design (forward hook on last ViT block, CLS-to-patch attention, gamma=0.6, brightness-modulated output instead of false-color heatmap) |
| `src/app.py` | Flask routes, model loading, base64 transport of brightness map | Multi-model architecture decision, port 5001 workaround for macOS AirPlay conflict, decision to run all three models per request |
| `frontend/src/App.jsx` | Full UI including dark Duke theme, tab switcher, animated confidence bars, attention brightness map display | Design direction (dark/dramatic), content of landmark facts, model description text, all visual style decisions, decision to show brightness map instead of uploaded image preview |
| `data/descriptions/landmarks.json` | Draft paragraph descriptions for each landmark | All editing and refinement of visual descriptions to front-load distinctive features within CLIP's 77-token limit |
| `SETUP.md`, `README.md` | Document templates and structure | All accuracy numbers, experiment results, real-world framing, video links |

**What AI did not do:** The experiment design (which ablations to run, which architectures to compare), all training runs (executed on Google Colab T4 GPU), the iteration log, the data collection, and the analysis conclusions are entirely Alexander Zarboulas's original work.

---

## Libraries and Frameworks

| Library | Use |
|---------|-----|
| PyTorch + torchvision | Training loop, data loading, ViT model |
| openai/clip | CLIP model loading, image and text encoding |
| scikit-learn | Stratified train/val/test split, classification report |
| matplotlib + seaborn | Training curves (loss and accuracy per epoch) |
| numpy | Attention weight reshaping and brightness map pixel arithmetic |
| Flask + flask-cors | REST API backend |
| React + Vite | Frontend UI |
| Inter + Playfair Display (Google Fonts) | UI typography |
| Pillow | Image decoding and brightness map generation in inference |

---

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Arandjelovic et al. (2016). *NetVLAD: CNN architecture for weakly supervised place recognition.* CVPR 2016.
- PyTorch documentation: https://pytorch.org/docs/stable/
- OpenAI CLIP repository: https://github.com/openai/CLIP
