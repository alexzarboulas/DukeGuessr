#!/usr/bin/env bash
# One-time setup for DukeGuessr.
# Run once after cloning the repo: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "==> Creating Python virtual environment..."
python3 -m venv venv

echo "==> Installing Python dependencies..."
source venv/bin/activate
pip install --quiet -r requirements.txt
pip install --quiet git+https://github.com/openai/CLIP.git

echo "==> Installing frontend dependencies..."
cd frontend && npm install --silent && cd ..

echo "==> Downloading model weights..."
mkdir -p models
curl -L --progress-bar \
  https://github.com/alexzarboulas/DukeGuessr/releases/download/v1.0/clip_base_best.pth \
  -o models/clip_base_best.pth
curl -L --progress-bar \
  https://github.com/alexzarboulas/DukeGuessr/releases/download/v1.0/vit_base_best.pth \
  -o models/vit_base_best.pth

echo ""
echo "Setup complete. Run: bash start.sh"
