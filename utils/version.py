"""
utils/version.py

DeepAxon version information.
Single source of truth — imported by logger and any other module that needs it.
Do not put version in config.json (user-editable) or hardcode it elsewhere.
"""
import sys

__version__ = "5.1.0"
__codename__ = "v5_analysis"

def get_env_info():
    info = {
        "python": sys.version.split()[0],
    }

    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
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

    return info