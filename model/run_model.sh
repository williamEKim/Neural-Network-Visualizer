#!/bin/bash
set -e

if ! command -v python3 &>/dev/null; then
    echo "python3 not found. Install it from https://www.python.org or via Homebrew."
    exit 1
fi

echo "Initiating Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

unalias python 2>/dev/null || true

echo "Installing all dependencies..."
pip install -r requirements.txt

echo "Installation complete! Executing model training..."
python3 train.py