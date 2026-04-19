# Setup Instructions

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm

---

## 1. Clone the repo

```bash
git clone https://github.com/alexzarboulas/DukeGuessr.git
cd DukeGuessr
```

---

## 2. One-time setup

```bash
bash setup.sh
```

This installs Python dependencies, the CLIP library, frontend packages, and downloads both model weight files (~900 MB total). It only needs to be run once.

---

## 3. Start the app

```bash
bash start.sh
```

Both the backend and frontend start in the same terminal. Open the URL printed in the terminal (typically http://localhost:3000). Press **Ctrl+C** to stop both servers.

---

## macOS note

AirPlay occupies port 5000. The backend defaults to port 5001 — no action needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: clip` | Run `bash setup.sh` again — the CLIP install may have failed |
| SSL error downloading CLIP weights | `curl -L https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -o ~/.cache/clip/ViT-B-32.pt` |
| Frontend not on port 3000 | Vite auto-increments if 3000 is taken — check the terminal output for the actual URL |
| `venv/bin/activate: No such file` | Run `bash setup.sh` first to create the virtual environment |
