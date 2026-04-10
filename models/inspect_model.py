"""
inspect_model.py

Print embedded metadata from a DeepAxon .pt model file.
Usage (from models/ folder) : python inspect_model.py rabbit_40x_v1.pt
Usage (from repo root)      : python models/inspect_model.py rabbit_40x_v1.pt
"""

import sys
import torch
from pathlib import Path

def inspect(model_path: str):
    model_path = Path(model_path).resolve()
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print("Legacy model format — no embedded metadata.")
        print(f"Keys found: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'raw state dict'}")
        return

    meta = checkpoint.get('meta', {})
    if not meta:
        print("v5 format detected but no metadata found.")
        return

    # ── Print metadata ────────────────────────────────────────────────
    width = max(len(k) for k in meta.keys())
    print(f"\n{'=' * 60}")
    print(f"  {Path(model_path).name}")
    print(f"{'=' * 60}")

    sections = {
        'Identity':        ['model_name', 'version', 'codename', 'trained_date'],
        'Architecture':    ['architecture', 'encoder', 'encoder_weights', 'in_channels',
                            'classes', 'input_size', 'activation', 'normalization'],
        'Dataset':         ['magnification', 'patch_size', 'dataset_path', 'split_mode',
                            'val_images', 'n_train_patches', 'n_val_patches'],
        'Results':         ['best_epoch', 'best_axon_dice', 'best_myelin_dice',
                            'best_val_loss', 'epochs_completed', 'early_stopped'],
        'Training config': ['augmentation', 'geo_prob', 'photo_prob', 'batch_size',
                            'epochs_limit', 'learning_rate', 'dice_weight', 'ce_weight',
                            'ce_smooth', 'reduce_lr_patience', 'early_stop_patience',
                            'early_stop_min_delta'],
        'Environment':     ['gpu', 'torch_version', 'python_version', 'hostname'],
    }

    for section, keys in sections.items():
        print(f"\n  ── {section} {'─' * (54 - len(section))}")
        for k in keys:
            if k in meta:
                v = meta[k]
                if isinstance(v, float):
                    v = f"{v:.4f}"
                elif isinstance(v, list):
                    v = ', '.join(str(x) for x in v) if v else '(none)'
                print(f"  {k:<{width}} : {v}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py path/to/model.pt")
        sys.exit(1)
    inspect(sys.argv[1])