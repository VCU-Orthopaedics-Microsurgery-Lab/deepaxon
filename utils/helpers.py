"""
utils/helpers.py

Shared utility functions used across segment, morphometrics, train, and batch_axon.

Note: keys prefixed with '_section_' in config.json are decorative dividers
and are safely ignored by all .get() calls in this file.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────

_CONFIG_PATH  = Path(__file__).resolve().parent.parent / "config.json"
_config_cache = None


def load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"config.json not found at {_CONFIG_PATH}")
        with open(_CONFIG_PATH, 'r') as f:
            _config_cache = json.load(f)
    return _config_cache


def get_pixel_size(mag: str, image_width: int) -> float | None:
    """
    Return µm/pixel for a given magnification and image width.
    Returns None if not configured (e.g. 100X placeholder).
    mag: '40X', '100X', etc.
    image_width: actual pixel width of the image
    """
    config = load_config()
    sizes  = config.get("pixel_size_um", {}).get(mag)
    if sizes is None:
        return None
    return sizes.get(str(image_width))


def get_fiji_path() -> str:
    """
    Return Fiji executable path from config.json.
    Prompts user and saves to config.json if not set.
    Also updates the in-memory cache so the value is available immediately.
    """
    global _config_cache
    config    = load_config()
    fiji_path = config.get("fiji_executable", "").strip()
    if not fiji_path:
        fiji_path = input("Enter the path to your Fiji executable: ").strip()
        config["fiji_executable"] = fiji_path
        # Update cache before writing so mid-session calls reflect the new path
        if _config_cache is not None:
            _config_cache["fiji_executable"] = fiji_path
        with open(_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
    return fiji_path


# ─── Study / folder scanning ──────────────────────────────────────────────────

def _has_images(d: Path, source_dir: Path = None) -> bool:
    """
    Return True if directory d contains image files.
    If source_dir is provided, checks that all source image stems have
    a corresponding segmented file in d.
    """
    if not d.exists():
        return False
    imgs = [f for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in ('.tif', '.tiff', '.png')]
    if not imgs:
        return False
    if source_dir is not None and source_dir.exists():
        config         = load_config()
        seg_suffix     = config.get("segmented_suffix", "_segmented")
        source_stems   = {f.stem for f in source_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in ('.tif', '.tiff')}
        segmented_stems = {f.stem.replace(seg_suffix, '') for f in imgs}
        return source_stems.issubset(segmented_stems)
    return True


def scan_study(study_dir: str, ignore_4x: bool = True) -> dict:
    """
    Walk a study directory and return a dict:
    {
        animal_name: {
            nerve_name: {
                'mag': '40X',
                'tiff_dir': Path,
                'segmented_dir': Path | None,
                'morphometrics_dir': Path | None,
                'csa_dir': Path | None,
            }
        }
    }
    Ignores 4X_TIFF folders if ignore_4x=True.
    """
    config        = load_config()
    tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
    seg_folder    = config.get("segmented_folder", "Segmented")
    morph_folder  = config.get("morphometrics_folder", "Morphometrics")
    csa_folder    = config.get("csa_folder", "CSA")

    study_path = Path(study_dir)
    result     = {}

    for animal_path in sorted(study_path.iterdir()):
        if not animal_path.is_dir():
            continue
        animal_name          = animal_path.name
        result[animal_name]  = {}

        for nerve_path in sorted(animal_path.iterdir()):
            if not nerve_path.is_dir():
                continue
            nerve_name = nerve_path.name

            # Find magnification TIFF folder
            mag      = None
            tiff_dir = None
            for sub in nerve_path.iterdir():
                matched_suffix = next((s for s in tiff_suffixes if sub.name.upper().endswith(s)), None)
                if sub.is_dir() and matched_suffix:
                    detected_mag = sub.name.upper().replace(matched_suffix, "")
                    if ignore_4x and detected_mag == "4X":
                        continue
                    mag      = detected_mag
                    tiff_dir = sub
                    break

            if tiff_dir is None:
                continue

            seg_dir   = nerve_path / seg_folder
            morph_dir = nerve_path / morph_folder
            csa_dir   = nerve_path / csa_folder

            result[animal_name][nerve_name] = {
                'mag':              mag,
                'tiff_dir':         tiff_dir,
                'segmented_dir':    seg_dir   if _has_images(seg_dir, tiff_dir) else None,
                'morphometrics_dir':morph_dir if _has_images(morph_dir)         else None,
                'csa_dir':          csa_dir   if csa_dir.exists()               else None,
            }

    return result


def detect_input_level(path: str) -> str:
    """
    Detect whether a path is at study, animal, nerve, or tiff level.
    Returns: 'study', 'animal', 'nerve', or 'tiff'
    """
    config        = load_config()
    tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
    p             = Path(path)

    def has_tiff(folder):
        return any(
            sub.is_dir() and any(sub.name.upper().endswith(s) for s in tiff_suffixes)
            for sub in folder.iterdir()
        )

    if any(p.name.upper().endswith(s) for s in tiff_suffixes):
        return 'tiff'
    if has_tiff(p):
        return 'nerve'
    subdirs = [s for s in p.iterdir() if s.is_dir()]
    if subdirs and any(has_tiff(s) for s in subdirs):
        return 'animal'
    return 'study'


def resolve_scan(input_dir: str) -> tuple:
    """
    Accepts study, animal, nerve, or tiff level input.
    Returns (study_dict, study_dir) in the same format scan_study produces.
    """
    level  = detect_input_level(input_dir)
    p      = Path(input_dir)
    config = load_config()

    # Guard: reject known output folder names
    reserved = {
        config.get("segmented_folder",    "Segmented").lower(),
        config.get("morphometrics_folder","Morphometrics").lower(),
        config.get("csa_folder",          "CSA").lower(),
    }
    if p.name.lower() in reserved:
        raise ValueError(
            f"'{p.name}' is an output subfolder — please pass the nerve, animal, or study folder instead."
        )

    if level == 'tiff':
        nerve_path    = p.parent
        animal_path   = nerve_path.parent
        study_dir     = str(animal_path.parent)
        seg_folder    = config.get("segmented_folder",    "Segmented")
        morph_folder  = config.get("morphometrics_folder","Morphometrics")
        csa_folder    = config.get("csa_folder",          "CSA")
        tiff_suffixes = [s.upper() for s in config.get("tiff_suffixes", ["_TIFF"])]
        matched       = next((s for s in tiff_suffixes if p.name.upper().endswith(s)), "_TIFF")
        mag           = p.name.upper().replace(matched, "")
        nerve_data    = {
            'mag':              mag,
            'tiff_dir':         p,
            'segmented_dir':    None,  # always reprocess when TIFF folder passed directly
            'morphometrics_dir':p.parent / morph_folder if (p.parent / morph_folder).exists() else None,
            'csa_dir':          p.parent / csa_folder   if (p.parent / csa_folder).exists()   else None,
        }
        return {animal_path.name: {nerve_path.name: nerve_data}}, study_dir

    if level == 'nerve':
        animal_path = p.parent
        study_dir   = str(animal_path.parent)
        full        = scan_study(study_dir)
        return {
            animal_path.name: {
                p.name: full.get(animal_path.name, {}).get(p.name, {})
            }
        }, study_dir

    elif level == 'animal':
        study_dir = str(p.parent)
        full      = scan_study(study_dir)
        return {p.name: full.get(p.name, {})}, study_dir

    else:  # study
        return scan_study(input_dir), input_dir


def detect_study_mag(study_result: dict) -> str | None:
    """Detect the dominant magnification across a study (ignoring 4X)."""
    mags = [
        nerve.get('mag')
        for animal in study_result.values()
        for nerve in animal.values()
        if nerve.get('mag')
    ]
    if not mags:
        return None
    return max(set(mags), key=mags.count)


def list_models(models_dir: str = None) -> list[Path]:
    """Return list of .keras files in the models directory."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    else:
        models_dir = Path(models_dir)
    return sorted(models_dir.glob("*.keras"))


def list_files(directory: str, extensions: tuple = ('.tif', '.tiff', '.png')) -> list[Path]:
    """Return sorted list of image files in a directory."""
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def count_patches(directory: str) -> int:
    return len(list_files(directory, extensions=('.png', '.tif', '.tiff')))


def get_training_dir(images_dir: str) -> Path:
    """Return the training root (parent of images dir)."""
    return Path(images_dir).resolve().parent


def get_model_dir(images_dir: str) -> Path:
    return get_training_dir(images_dir) / "models"


def get_log_dir(images_dir: str) -> Path:
    return get_training_dir(images_dir) / "logs"


def center_crop(img: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Crop image to the largest dimensions divisible by patch_size, centered.
    Used by both segment/segment.py and train/data/preprocess.py.
    """
    h, w    = img.shape[:2]
    crop_h  = (h // patch_size) * patch_size
    crop_w  = (w // patch_size) * patch_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


# ─── User input helpers ───────────────────────────────────────────────────────

def get_int_input(prompt: str, default: int = None, min_val: int = 1) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == '' and default is not None:
            return default
        try:
            val = int(raw)
            if val >= min_val:
                return val
            print(f"  Please enter a value ≥ {min_val}.")
        except ValueError:
            print("  Please enter a valid integer.")


def get_float_input(prompt: str, default: float = None, min_val: float = 0.0, max_val: float = 1.0) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == '' and default is not None:
            return default
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a valid number.")


def get_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    raw  = input(f"{prompt} {hint}: ").strip().lower()
    if raw == '':
        return default
    return raw in ('y', 'yes')


# ─── Training helpers ─────────────────────────────────────────────────────────

def compute_batch_size(n_patches: int) -> int:
    """Return largest power-of-2 batch size that divides n_patches evenly, max 32."""
    for bs in [32, 16, 8, 4, 2]:
        if n_patches % bs == 0:
            return bs
    warnings.warn(
        f"No power-of-2 batch size divides {n_patches} evenly. "
        f"Defaulting to 4 — last batch may be smaller."
    )
    return 4


def compute_aug_prob(n_patches: int) -> float:
    """Return augmentation probability based on dataset size."""
    if n_patches < 100:
        return 0.5
    elif n_patches < 300:
        return 0.35
    return 0.25

def get_git_commit() -> str:
    """Return current git commit hash for model provenance tracking."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'