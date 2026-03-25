"""
utils/metrics.py

Loss functions and metrics for DeepAxon UNet++ training.
Tensor versions (TensorFlow/Keras) and numpy versions for evaluation.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# ─── Tensor metrics (used during training) ────────────────────────────────────

def dice_coef(y_true, y_pred, smooth: float = 1.0):
    """
    Dice similarity coefficient.
    smooth=1.0: Laplace smoothing — prevents division by zero on empty masks.
    Standard value for medical image segmentation.
    """
    y_true_f     = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth: float = 1.0):
    """
    Intersection over Union coefficient.
    smooth=1.0: Laplace smoothing — prevents division by zero on empty masks.
    """
    y_true_f     = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    union        = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    """
    Weighted combination of dice loss and binary cross-entropy.
    50/50 weighting: BCE improves pixel-level precision,
    Dice improves overlap quality. Weights configurable via
    config.json loss_weights if needed in future.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * K.mean(bce)


# ─── Numpy metrics (used for evaluation / logging) ────────────────────────────

def dice_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    """Dice coefficient for post-hoc evaluation. smooth=1.0 matches tensor version."""
    y_true_f     = y_true.flatten().astype(np.float32)
    y_pred_f     = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    """IoU coefficient for post-hoc evaluation. smooth=1.0 matches tensor version."""
    y_true_f     = y_true.flatten().astype(np.float32)
    y_pred_f     = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    union        = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)