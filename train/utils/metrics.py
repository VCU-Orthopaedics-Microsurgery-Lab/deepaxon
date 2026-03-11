# train/utils/metrics.py
"""
Segmentation metrics for DeepAxon training.

Contains:
- Keras tensor-based metrics for training and callbacks
- NumPy-based metrics for post-training evaluation and reporting
"""

# ------------------------------ Keras/TensorFlow Metrics --------------------------- #
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np

# ------------------------------ Training Metrics (tensor-based) ------------------- #
def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient (tensor) for training and callbacks"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient"""
    return 1 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1e-6):
    """Intersection over Union (IoU) coefficient (tensor)"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    """Combined categorical cross-entropy + Dice loss"""
    bce = CategoricalCrossentropy()(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc

# ------------------------------ Post-training Metrics (NumPy-based) ---------------- #
def dice_np(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for NumPy arrays (post-training evaluation)"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_np(y_true, y_pred, smooth=1e-6):
    """IoU score for NumPy arrays (post-training evaluation)"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)