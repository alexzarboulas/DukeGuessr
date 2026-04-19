#!/usr/bin/env bash
# Start the DukeGuessr backend and frontend in one terminal.
# Usage: bash start.sh

set -e
cd "$(dirname "$0")"

# Activate virtualenv
source venv/bin/activate

# Kill both servers cleanly on Ctrl+C
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$FLASK_PID" "$VITE_PID" 2>/dev/null
  wait "$FLASK_PID" "$VITE_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

echo "==> Starting backend (port 5001)..."
python src/app.py \
  --clip_weights  models/clip_base_best.pth \
  --vit_weights   models/vit_base_best.pth \
  --descriptions  data/descriptions/landmarks.json \
  --port 5001 &
FLASK_PID=$!

echo "==> Starting frontend..."
cd frontend && npm run dev &
VITE_PID=$!
cd ..

echo ""
echo "Both servers running. Open the URL shown above."
echo "Press Ctrl+C to stop."
echo ""

# Wait for either process to exit
wait "$FLASK_PID" "$VITE_PID"
