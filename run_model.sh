#!/bin/bash
set -e

CURR_DIR="$PWD"

if ! command -v python3 &>/dev/null; then
    echo "python3 not found. Install it from https://www.python.org or via Homebrew."
    exit 1
fi

echo "Initiating Virtual Environment..."
python3 -m venv "$CURR_DIR/venv"
source "$CURR_DIR/venv/bin/activate"

unalias python 2>/dev/null || true

echo "Installing all dependencies..."
pip install -r "$CURR_DIR/model/requirements.txt"

echo "Installation complete! Executing model training..."
python3 "$CURR_DIR/model/train.py"

echo "Model Training Complete! Now executing export.py ..."
python3 "$CURR_DIR/model/export.py"