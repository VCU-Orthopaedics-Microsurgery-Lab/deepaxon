"""
utils/version.py

DeepAxon version information.
Single source of truth — imported by logger and any other module that needs it.
Do not put version in config.json (user-editable) or hardcode it elsewhere.

Usage:
    python utils/version.py   — print version and full environment info
"""

import sys
import platform
import socket

__version__     = "5.1.0"
__codename__    = "v5_analysis"
__version_str__ = f"DeepAxon v{__version__} ({__codename__})"


def get_env_info() -> dict:
    """
    Collect full environment fingerprint.
    Used by logger header and version CLI.
    Sufficient for exact environment reproduction from a log file alone.
    """
    info = {
        "python":   sys.version.split()[0],
        "platform": platform.system(),
        "hostname": socket.gethostname(),
    }

    try:
        import torch
        info["pytorch"]        = torch.__version__
        info["cuda"]           = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["pytorch"] = None

    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError:
        info["numpy"] = None

    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None

    try:
        import segmentation_models_pytorch as smp
        info["smp"] = smp.__version__
    except ImportError:
        info["smp"] = None

    try:
        import monai
        info["monai"] = monai.__version__
    except ImportError:
        info["monai"] = None

    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = None

    try:
        import skimage
        info["scikit_image"] = skimage.__version__
    except ImportError:
        info["scikit_image"] = None

    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except ImportError:
        info["scikit_learn"] = None

    try:
        import timm
        info["timm"] = timm.__version__
    except ImportError:
        info["timm"] = None

    try:
        import PIL
        info["pillow"] = PIL.__version__
    except ImportError:
        info["pillow"] = None

    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except ImportError:
        info["pandas"] = None

    try:
        import patchify
        info["patchify"] = patchify.__version__
    except ImportError:
        info["patchify"] = None

    return info


def main() -> None:                          
    print(f"\n{__version_str__}\n")          
    for k, v in get_env_info().items():      
        status = str(v) if v is not None else "NOT FOUND"  
        print(f"  {k:<30} {status}")         
    print()                                  


if __name__ == "__main__":                   
    main()                                   