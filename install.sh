#!/bin/bash
# DeepAxon v5.1.0 — Environment Setup
# Branch: v5_analysis | Python: 3.11.x
#
# Usage (Linux/macOS):
#   chmod +x install.sh && ./install.sh
#
# Usage (Windows):
#   bash install.sh
#
# Prerequisites: Python 3.11.x virtual environment created and activated.

set -e  # Exit on any error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DeepAxon v5.1.0 — Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1 — Core dependencies ────────────────────────────────────────────────
echo ""
echo "[1/4] Installing core dependencies..."
pip install -r requirements.txt

# ── Step 2 — PyTorch CUDA wheel ───────────────────────────────────────────────
echo ""
echo "[2/4] Installing PyTorch 2.5.1 (cu121)..."
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 3 — Patchify ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] Installing patchify (--no-deps)..."
pip install patchify==0.2.3 --no-deps

# ── Step 4 — Verify ───────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
python utils/version.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installation complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"