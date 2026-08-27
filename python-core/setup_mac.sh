#!/bin/bash
set -euo pipefail

# OpenAutoNote setup for macOS (default cloud STT; no Streamlit).

cd "$(dirname "$0")"

echo "Starting OpenAutoNote setup..."

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is not installed."
    echo "Please install it using Homebrew: brew install ffmpeg"
    exit 1
fi
echo "ffmpeg found."

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed."
    exit 1
fi

echo "Installing Python dependencies (default cloud STT stack)..."
python3 -m pip install -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created python-core/.env from .env.example — edit it and set DASHSCOPE_API_KEY."
fi

echo
echo "Setup complete."
echo "  1. Put your DashScope key in python-core/.env (DASHSCOPE_API_KEY=sk-...)"
echo "  2. Run: python3 main.py"
echo "  3. Open: http://127.0.0.1:8964"
echo
echo "Optional local Whisper (Apple Silicon): pip install -r requirements_mac.txt"
