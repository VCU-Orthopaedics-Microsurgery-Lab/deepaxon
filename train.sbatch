#!/bin/bash
#SBATCH --job-name=deepaxon_train
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
source /lustre/home/mazurm/deepaxon/da_venv/bin/activate

# ── Run training (non-interactive via config file) ─────────────────────────────
python -m train --config train_config.json
