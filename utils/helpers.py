"""
utils/helpers.py

Shared utility functions used across segment, morphometrics, train, and batch_axon.

Note: keys prefixed with '_section_' in config.json are decorative dividers
and are safely ignored by all .get() calls in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import re

# ─── Config ───────────────────────────────────────────────────────────────────

_CONFIG_PATH  = Path(__file__).resolve().parent.parent / "config.json"
_config_cache = None

def natural_sort_key(path) -> list:
    """
    Natural sort key for filenames containing numbers.
    Sorts img_1, img_2 ... img_10 correctly instead of img_1, img_10, img_2.
    """
    parts = re.split(r'(\d+)', Path(path).stem)
    return [int(p) if p.isdigit() else p for p in parts]

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
        config          = load_config()
        seg_suffix      = config.get("segmented_suffix", "_segmented")
        source_stems    = {f.stem for f in source_dir.iterdir()
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
    seg_folder    = config.get("segmented_folder",    "Segmented")
    morph_folder  = config.get("morphometrics_folder","Morphometrics")
    csa_folder    = config.get("csa_folder",          "CSA")

    study_path = Path(study_dir)
    result     = {}

    for animal_path in sorted(study_path.iterdir()):
        if not animal_path.is_dir():
            continue
        animal_name         = animal_path.name
        result[animal_name] = {}

        for nerve_path in sorted(animal_path.iterdir()):
            if not nerve_path.is_dir():
                continue
            nerve_name = nerve_path.name

            mag      = None
            tiff_dir = None
            for sub in nerve_path.iterdir():
                matched_suffix = next(
                    (s for s in tiff_suffixes if sub.name.upper().endswith(s)), None
                )
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
                'mag':               mag,
                'tiff_dir':          tiff_dir,
                'segmented_dir':     seg_dir   if _has_images(seg_dir, tiff_dir) else None,
                'morphometrics_dir': morph_dir if _has_images(morph_dir)         else None,
                'csa_dir':           csa_dir   if csa_dir.exists()               else None,
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
            'mag':               mag,
            'tiff_dir':          p,
            'segmented_dir':     None,
            'morphometrics_dir': p.parent / morph_folder if (p.parent / morph_folder).exists() else None,
            'csa_dir':           p.parent / csa_folder   if (p.parent / csa_folder).exists()   else None,
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

    else:
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
    """Return list of .pt model files in the models directory."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    else:
        models_dir = Path(models_dir)
    return sorted(models_dir.glob("*.pt"))


def list_files(directory: str, extensions: tuple = ('.tif', '.tiff', '.png')) -> list[Path]:
    """Return sorted list of image files in a directory."""
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def count_patches(directory: str) -> int:
    return len(list_files(directory, extensions=('.png', '.tif', '.tiff', '.bmp')))


def get_training_dir(images_dir: str) -> Path:
    """Return the training root (parent of images dir)."""
    return Path(images_dir).resolve().parent


def get_model_dir(images_dir: str) -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def get_log_dir(images_dir: str) -> Path:
    return Path(__file__).resolve().parent.parent / "logs"


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


def get_float_input(
    prompt: str,
    default: float = None,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> float:
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

def _get_gpu_batch_candidates() -> list[int] | None:
    """
    Return batch size candidates based on available GPU VRAM.
    VRAM-based detection works for any GPU without name matching.
    Returns None if no GPU available or detection fails.

    Tiers:
    ≥ 60GB  : A100 80GB, H100, H200          → [512, 256, 128, 64, 32]
    ≥ 20GB  : A100 40GB, A40, RTX 3090/4090  → [256, 128, 64, 32, 16]
    ≥ 10GB  : V100 16GB, RTX 3080, RTX 4070  → [128, 64, 32, 16, 8]
    ≥ 6GB   : RTX 3060, GTX 1080             → [64, 32, 16, 8, 4]
    < 6GB   : anything smaller               → [16, 8, 4, 2]
    """
    try:
        import torch    # local import — torch not needed at module level in helpers

        if not torch.cuda.is_available():
            return None
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if  vram_gb >= 60:
            return [512, 256, 128, 64, 32]
        elif vram_gb >= 20:
            return [256, 128, 64, 32, 16]
        elif vram_gb >= 10:
            return [128, 64, 32, 16, 8]
        elif vram_gb >= 6:
            return [64, 32, 16, 8, 4]
        else:
            return [16, 8, 4, 2]
    except Exception:
        return None

def _get_ideal_batch_sizes(use_gpu: bool) -> list[int]:
    if use_gpu:
        candidates = _get_gpu_batch_candidates()
        if candidates:
            return candidates  # all candidates for this VRAM tier
    config    = load_config()
    train_cfg = config.get("training", {})
    cpu_cands = train_cfg.get("cpu_batch_candidates", [32, 16, 8, 4])
    return cpu_cands


def _classify_batch(bs: int, n_patches: int, min_ok: float, min_warn: float) -> tuple[str, int]:
    """
    Classify a batch size candidate into a zone.

    Returns:
        (zone, remainder)
        zone: 'perfect' | 'acceptable' | 'excluded' | 'danger' | 'skip'
        remainder: patches in last batch (0 = perfect fit)

    Zones:
        perfect    remainder == 0          — even division, use immediately
        acceptable remainder/bs >= min_ok  — last batch sufficiently full
        excluded   min_warn <= r/bs < min_ok — middle ground, skip
        danger     remainder/bs < min_warn  — drop remainder, treat as perfect fit
        skip       n_patches < bs * 2      — not enough patches for 2 full batches
    """
    if n_patches < bs * 2:
        return 'skip', 0
    remainder = n_patches % bs
    if remainder == 0:
        return 'perfect', 0
    fullness = remainder / bs
    if fullness >= min_ok:
        return 'acceptable', remainder
    elif fullness >= min_warn:
        return 'excluded', remainder
    else:
        return 'danger', remainder


def compute_batch_options(
    n_patches: int,
    use_gpu: bool = False
) -> dict:
    """
    Evaluate all candidate batch sizes against three-zone logic and
    return a structured result for menu display and selection.

    Three zones:
        ≥ 80% full   → acceptable  — present as menu option (power-of-2 candidates)
        15–80% full  → excluded    — not recommended
        < 15% full   → danger      — drop remainder, present as trim option

    Perfect fit divisors (non power-of-2) are always shown as a separate section.

    Args:
        n_patches: estimated training patches (after val split)
        use_gpu:   True if training on GPU

    Returns dict with keys:
        'acceptable'   : list of (bs, remainder) — last batch ≥80% full
        'perfect_fits' : list of int — exact divisors of n_patches, non power-of-2
        'trim'         : list of (bs, n_dropped, pct_dropped) — <15%, drop remainder
        'excluded'     : list of (bs, remainder, fullness_pct) — 15-80%, shown as info
        'ideal'        : list[int] — all device batch sizes regardless of patches
        'device_label' : str — e.g. 'NVIDIA H100 80GB HBM3 (85GB VRAM)' or 'CPU'
    """
    config    = load_config()
    train_cfg = config.get("training", {})
    min_ok    = train_cfg.get("min_last_batch_fullness",    0.80)
    min_warn  = train_cfg.get("danger_last_batch_fullness", 0.15)

    if use_gpu:
        candidates = _get_gpu_batch_candidates()
        if candidates is None:
            candidates = train_cfg.get("gpu_batch_candidates", [512, 256, 128, 64, 32])
    else:
        candidates = train_cfg.get("cpu_batch_candidates", [32, 16, 8, 4])

    acceptable = []
    trim       = []
    excluded   = []

    for bs in candidates:
        zone, remainder = _classify_batch(bs, n_patches, min_ok, min_warn)

        if zone == 'skip':
            continue
        elif zone in ('perfect', 'acceptable'):
            acceptable.append((bs, remainder))
        elif zone == 'excluded':
            fullness_pct = int(remainder / bs * 100)
            excluded.append((bs, remainder, fullness_pct))
        elif zone == 'danger':
            n_dropped   = remainder
            pct_dropped = round(n_dropped / n_patches * 100, 1)
            trim.append((bs, n_dropped, pct_dropped))

    # ── Perfect fit divisors (non power-of-2) ─────────────────────────────────
    # Largest 3 non-power-of-2 exact divisors in range [16, n_patches//2]
    def _is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    perfect_fits = sorted(
        [bs for bs in range(16, n_patches // 2 + 1)
         if n_patches % bs == 0 and not _is_power_of_2(bs)],
        reverse=True
    )[:3]

    if use_gpu:
        try:
            import torch
            vram_gb      = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name     = torch.cuda.get_device_name(0)
            device_label = f"{gpu_name} ({vram_gb:.0f}GB VRAM)"
        except Exception:
            device_label = "GPU (VRAM unknown)"
    else:
        device_label = "CPU"

    return {
        'acceptable':   acceptable,
        'perfect_fits': perfect_fits,
        'trim':         trim,
        'excluded':     excluded,
        'ideal':        _get_ideal_batch_sizes(use_gpu),
        'device_label': device_label,
    }
    
    
def get_hann_compatible_step(patch_size: int) -> int:
    """
    Return the patch step size compatible with Hann window blending.
    50% overlap (step = patch_size // 2) is required — the 9-position
    Hann grid in segment.py assumes each pixel is covered by exactly
    4 patches. Changing this breaks the blending math.
    """
    return patch_size // 2