# train/utils/logger.py
"""
Simple logging utility to write messages to file and console
"""

import logging
import os

def setup_logger(log_file=None, level=logging.INFO):
    """Setup logger that logs to console and optionally to a file"""
    logger = logging.getLogger("DeepAxon")
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger