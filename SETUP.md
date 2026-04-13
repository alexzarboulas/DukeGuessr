# Setup Instructions

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm

---

## 1. Clone the repo

```bash
git clone <repo-url>
cd DukeGuessr
```

---

## 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

---

## 3. Download model weights

The weights are too large for GitHub (CLIP: 577 MB, ViT: 327 MB) and are hosted as GitHub Release assets.
Download the two required files into the `models/` folder:

```bash
curl -L https://github.com/alexzarboulas/DukeGuessr/releases/download/v1.0/clip_base_best.pth -o models/clip_base_best.pth
curl -L https://github.com/alexzarboulas/DukeGuessr/releases/download/v1.0/vit_base_best.pth  -o models/vit_base_best.pth
```

> Full release page: https://github.com/alexzarboulas/DukeGuessr/releases/tag/v1.0

---

## 4. Start the backend

```bash
# From project root, with venv active
python src/app.py \
  --clip_weights models/clip_base_best.pth \
  --vit_weights  models/vit_base_best.pth \
  --descriptions data/descriptions/landmarks.json \
  --port 5001
```

> **macOS:** AirPlay occupies port 5000 — use `--port 5001` (already the default above).

---

## 5. Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Create `frontend/.env.local` with:
```
VITE_API_URL=http://localhost:5001
```

Then open the URL shown in the terminal (typically http://localhost:3000).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: clip` | Run `pip install git+https://github.com/openai/CLIP.git` |
| SSL error downloading CLIP base weights | `curl -L https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -o ~/.cache/clip/ViT-B-32.pt` |
| Port 5000 returns 403 on macOS | AirPlay is on 5000 — use `--port 5001` |
| CORS error in browser | Check that `VITE_API_URL` in `frontend/.env.local` matches the Flask port |
| Vite not on port 3000 | Vite auto-increments if 3000 is taken — check terminal output for actual URL |
