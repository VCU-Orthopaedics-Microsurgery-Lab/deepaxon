"""
batch_axon/overlay/process_overlay.py

Processes CSA overlay images created in Fiji.
Opens each image in Fiji headless, exports the overlay ROI,
computes the pixel area of the traced region, then cleans up.

Fiji executable path is stored in config.json.
"""

from __future__ import annotations

import os
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import roifile

from utils.helpers import get_fiji_path

# Path to the Fiji macro — relative to repo root
_MACRO_PATH = Path(__file__).resolve().parent / "export_roi.ijm"


def run_fiji_export(img_path: str, roi_path: str) -> bool:
    """
    Run the Fiji macro to export the overlay ROI from an image.

    Args:
        img_path: path to the overlay TIFF
        roi_path: path where the .roi file will be saved

    Returns:
        True on success, False on failure.
    """
    fiji_path = get_fiji_path()

    if Path(roi_path).exists():
        Path(roi_path).unlink()

    command_args = f"{img_path},{roi_path}"
    command = [fiji_path, "-batch", str(_MACRO_PATH), command_args]

    try:
        subprocess.run(command, check=True, timeout=90, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"[WARN] Fiji timed out for: {img_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Fiji returned error for {img_path}: {e}")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Fiji executable not found: {fiji_path}")
        print("        Update the 'fiji_executable' field in config.json")
        return False

    if not Path(roi_path).exists():
        print(f"[WARN] ROI file not created for: {img_path}")
        return False

    return True


def get_overlay_area(img_path: str) -> int | None:
    """
    Extract the pixel area of the CSA overlay from a Fiji overlay TIFF.

    Args:
        img_path: path to the _CSA.tif overlay file

    Returns:
        Pixel count of the traced area, or None on failure.
    """
    img_path = Path(img_path)
    roi_path = img_path.with_suffix('.roi')

    if not img_path.exists():
        print(f"[WARN] CSA overlay not found: {img_path}")
        return None

    success = run_fiji_export(str(img_path), str(roi_path))
    if not success:
        return None

    try:
        roi = roifile.ImagejRoi.fromfile(str(roi_path))
        with Image.open(str(img_path)) as img:
            image_width, image_height = img.size

        mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)
        coords = [(x, y) for x, y in roi.coordinates()]
        draw.polygon(coords, outline=1, fill=1)

        mask_array = np.array(mask)
        pixel_count = int(np.count_nonzero(mask_array))

        return pixel_count

    except Exception as e:
        print(f"[WARN] Failed to process ROI for {img_path.name}: {e}")
        return None

    finally:
        if roi_path.exists():
            roi_path.unlink()
            
            
def get_overlay_mask(img_path: str) -> np.ndarray | None:
    """
    Returns full binary ROI mask aligned to the CSA image.

    This is used for:
    - morphometrics masking
    - Jupyter notebook QC
    - segmentation filtering
    """
    img_path = Path(img_path)
    roi_path = img_path.with_suffix('.roi')

    if not img_path.exists():
        print(f"[WARN] CSA overlay not found: {img_path}")
        return None

    success = run_fiji_export(str(img_path), str(roi_path))
    if not success:
        return None

    try:
        roi = roifile.ImagejRoi.fromfile(str(roi_path))

        with Image.open(str(img_path)) as img:
            image_width, image_height = img.size

        mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)

        coords = [tuple(map(int, np.round(p))) for p in roi.coordinates()]

        draw.polygon(coords, outline=1, fill=1)

        mask_array = np.array(mask).astype(bool)

        return mask_array

    except Exception as e:
        print(f"[WARN] Failed to generate ROI mask for {img_path.name}: {e}")
        return None

    finally:
        if roi_path.exists():
            roi_path.unlink()