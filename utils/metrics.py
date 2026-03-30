"""
utils/metrics.py

Loss functions and metrics for DeepAxon UNet++ training.
All metrics handle 3-class softmax output (background, myelin, axon).

Tensor versions (TensorFlow/Keras) — used during training via model.compile()
Numpy versions — used for post-hoc evaluation in evaluate/ module.

Class convention (matches BGW contract in segment.py and morphometrics.py):
    Class 0 = background
    Class 1 = myelin
    Class 2 = axon
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# ─── Per-class dice (building block) ─────────────────────────────────────────

def _dice_coef_class(y_true, y_pred, class_idx: int, smooth: float = 1.0):
    """
    Dice coefficient for a single class.
    Extracts the class_idx channel from one-hot y_true and softmax y_pred.
    """
    y_true_c     = tf.cast(y_true[..., class_idx], tf.float32)
    y_pred_c     = tf.cast(y_pred[..., class_idx], tf.float32)
    y_true_f     = K.flatten(y_true_c)
    y_pred_f     = K.flatten(y_pred_c)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# ─── Tensor metrics (used during training) ────────────────────────────────────

def dice_coef(y_true, y_pred, smooth: float = 1.0):
    """
    Macro-averaged Dice coefficient across all 3 classes.
    Averages per-class dice: background(0), myelin(1), axon(2).
    smooth=1.0: Laplace smoothing — prevents division by zero on empty masks.

    Use as metric in model.compile(metrics=[dice_coef]).
    """
    d0 = _dice_coef_class(y_true, y_pred, 0, smooth)  # background
    d1 = _dice_coef_class(y_true, y_pred, 1, smooth)  # myelin
    d2 = _dice_coef_class(y_true, y_pred, 2, smooth)  # axon
    return (d0 + d1 + d2) / 3.0


def dice_coef_axon(y_true, y_pred, smooth: float = 1.0):
    """
    Dice coefficient for axon class only (class 2).
    Tracked separately during training to monitor axon segmentation quality.
    Use as metric in model.compile(metrics=[dice_coef, dice_coef_axon]).
    """
    return _dice_coef_class(y_true, y_pred, 2, smooth)


def dice_coef_myelin(y_true, y_pred, smooth: float = 1.0):
    """
    Dice coefficient for myelin class only (class 1).
    Tracked separately during training to monitor myelin segmentation quality.
    """
    return _dice_coef_class(y_true, y_pred, 1, smooth)


def dice_loss(y_true, y_pred):
    """
    Macro-averaged Dice loss — 1 minus macro Dice coefficient.
    Lower is better.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth: float = 1.0):
    """
    Macro-averaged IoU (Jaccard) coefficient across all 3 classes.
    smooth=1.0: Laplace smoothing — prevents division by zero on empty masks.
    """
    def _iou_class(class_idx):
        y_true_c     = tf.cast(y_true[..., class_idx], tf.float32)
        y_pred_c     = tf.cast(y_pred[..., class_idx], tf.float32)
        y_true_f     = K.flatten(y_true_c)
        y_pred_f     = K.flatten(y_pred_c)
        intersection = K.sum(y_true_f * y_pred_f)
        union        = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    return (_iou_class(0) + _iou_class(1) + _iou_class(2)) / 3.0


def combined_loss(y_true, y_pred):
    """
    Weighted combination of macro Dice loss and categorical cross-entropy.

    Uses categorical_crossentropy (not binary) — correct for 3-class softmax.
    50/50 weighting: CCE improves per-pixel class discrimination,
    Dice improves spatial overlap quality.

    Both components are macro-averaged across all 3 classes.
    """
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * K.mean(cce)


# ─── Numpy metrics (used for evaluation / logging) ────────────────────────────

def dice_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    smooth: float = 1.0,
    per_class: bool = False
) -> float | dict:
    """
    Macro-averaged Dice coefficient for post-hoc evaluation.

    Args:
        y_true:    (N, H, W, 3) one-hot or (H, W, 3) one-hot
        y_pred:    same shape — softmax probabilities or argmax predictions
        smooth:    Laplace smoothing — matches tensor version
        per_class: if True returns dict with per-class and mean dice

    Returns:
        float (mean dice) or dict with keys:
            'background', 'myelin', 'axon', 'mean'
    """
    scores = {}
    class_names = ['background', 'myelin', 'axon']

    for i, name in enumerate(class_names):
        y_true_c     = y_true[..., i].flatten().astype(np.float32)
        y_pred_c     = y_pred[..., i].flatten().astype(np.float32)
        intersection = np.sum(y_true_c * y_pred_c)
        scores[name] = float(
            (2.0 * intersection + smooth) /
            (np.sum(y_true_c) + np.sum(y_pred_c) + smooth)
        )

    scores['mean'] = float(np.mean([scores[n] for n in class_names]))

    if per_class:
        return scores
    return scores['mean']


def iou_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    smooth: float = 1.0,
    per_class: bool = False
) -> float | dict:
    """
    Macro-averaged IoU coefficient for post-hoc evaluation.

    Args:
        y_true:    (N, H, W, 3) one-hot
        y_pred:    same shape
        smooth:    Laplace smoothing — matches tensor version
        per_class: if True returns dict with per-class and mean IoU

    Returns:
        float (mean IoU) or dict with keys:
            'background', 'myelin', 'axon', 'mean'
    """
    scores = {}
    class_names = ['background', 'myelin', 'axon']

    for i, name in enumerate(class_names):
        y_true_c     = y_true[..., i].flatten().astype(np.float32)
        y_pred_c     = y_pred[..., i].flatten().astype(np.float32)
        intersection = np.sum(y_true_c * y_pred_c)
        union        = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
        scores[name] = float((intersection + smooth) / (union + smooth))

    scores['mean'] = float(np.mean([scores[n] for n in class_names]))

    if per_class:
        return scores
    return scores['mean']