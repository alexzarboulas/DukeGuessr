# Attribution

## Dataset

All 350 landmark photos were **collected originally** by Alexander Zarboulas using a personal camera during dedicated photography sessions at Duke University in April 2026. No third-party images were used. Photos are not shown in this repository.

**7 classes collected:** Perkins Library, Main Quad, Duke Chapel, Campus Dr / Bus Stop, Sarah P. Duke Gardens, Wannamaker Benches, Other. ~50 images per class.

---

## AI Development Tools

AI tools (Claude Code, Anthropic) were used throughout this project for code scaffolding, debugging, and frontend design. The following files were scaffolded or drafted with AI assistance:

| File | AI contribution |
|------|----------------|
| `src/data.py` | Dataset class, transform pipeline, split logic |
| `src/clip_model.py` | CLIP loading wrappers, text encoding, logit computation |
| `src/train_clip.py` | Training loop structure, early stopping class |
| `src/train_vit.py` | ViT training loop |
| `src/evaluate.py` | Evaluation pipeline structure, per-class metrics output |
| `src/predict.py` | Single-image inference, GPS lookup, brightness map generation |
| `src/app.py` | Flask routes, model loading, base64 transport of brightness map |
| `frontend/src/App.jsx` | Full UI including dark Duke theme, tab switcher, animated confidence bars, attention brightness map display |
| `data/descriptions/landmarks.json` | Draft paragraph descriptions for each landmark |
| `SETUP.md`, `README.md` | Document templates and structure |

**What AI was not responsible for:** The experiment design, choice of architectures, all training runs (executed on Google Colab T4 GPU), hyperparameter decisions, ablation design, the attention visualization concept and implementation strategy, the data collection, and all analysis and conclusions are entirely Alexander Zarboulas's original work. AI generated code on request — it did not direct the project, choose what to build, or determine what the results mean.

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
