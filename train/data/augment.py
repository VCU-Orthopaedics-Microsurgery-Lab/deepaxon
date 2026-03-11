# train/data/augment.py
"""
Lightweight, CPU-based data augmentation for DeepAxon.
Operates on batches of image/mask patches (NumPy arrays).
Masks only receive geometric transforms; images receive intensity transforms as well.
"""

import numpy as np
import cv2
import random

def augment_dataset_np(
    X,
    Y,
    aug_prob=0.25,
    p_flip_h=0.5,
    p_flip_v=0.5,
    p_rotate=0.3,
    p_brightness=0.25,
    p_gamma=0.2,
    p_noise=0.1
):
    """
    Perform lightweight, on-the-fly CPU augmentations on (X, Y) batches.

    Parameters:
        X (np.ndarray): Batch of images (H x W x C)
        Y (np.ndarray): Batch of masks (H x W x C)
        aug_prob (float): Probability that any augmentation is applied to a patch
        p_flip_h, p_flip_v, p_rotate, p_brightness, p_gamma, p_noise:
            Individual probabilities for each augmentation type

    Returns:
        X_out, Y_out, flags (np.ndarrays): Augmented images, masks, and boolean flags
    """
    X_out, Y_out, flags = [], [], []

    for img, mask in zip(X, Y):
        img_out = img.copy()
        mask_out = mask.copy()
        augmented = False

        if random.random() < aug_prob:
            # Horizontal flip
            if random.random() < p_flip_h:
                img_out = np.flip(img_out, axis=1)
                mask_out = np.flip(mask_out, axis=1)
                augmented = True

            # Vertical flip
            if random.random() < p_flip_v:
                img_out = np.flip(img_out, axis=0)
                mask_out = np.flip(mask_out, axis=0)
                augmented = True

            # Small rotation (-10° to +10°)
            if random.random() < p_rotate:
                angle = random.uniform(-10, 10)
                center = (img_out.shape[1] / 2, img_out.shape[0] / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1)

                img_out = cv2.warpAffine(
                    img_out, M, (img_out.shape[1], img_out.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )

                mask_out = cv2.warpAffine(
                    mask_out.astype(np.uint8), M, (mask_out.shape[1], mask_out.shape[0]),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
                )
                mask_out = np.clip(np.round(mask_out), 0, 2).astype(np.uint8)
                augmented = True

            # Brightness / contrast jitter (images only)
            if random.random() < p_brightness:
                alpha = random.uniform(0.95, 1.05)
                beta = random.uniform(-0.05, 0.05)
                img_out = np.clip(img_out * alpha + beta, 0.0, 1.0)
                augmented = True

            # Gamma adjustment (images only)
            if random.random() < p_gamma:
                gamma = random.uniform(0.95, 1.05)
                img_out = np.clip(img_out ** gamma, 0.0, 1.0)
                augmented = True

            # Tiny Gaussian noise (images only)
            if random.random() < p_noise:
                noise = np.random.normal(0, 0.01, img_out.shape)
                img_out = np.clip(img_out + noise, 0.0, 1.0)
                augmented = True

        # Ensure channel dimension
        if img_out.ndim == 2:
            img_out = img_out[..., np.newaxis]
        if mask_out.ndim == 2:
            mask_out = mask_out[..., np.newaxis]

        X_out.append(img_out)
        Y_out.append(mask_out)
        flags.append(augmented)

    return np.array(X_out, dtype=np.float32), np.array(Y_out, dtype=np.uint8), np.array(flags, dtype=bool)


def augment_pair(img, mask):
    """
    Convenience wrapper for a single image/mask pair.
    Converts to batch of 1, calls `augment_dataset_np`, returns first element.
    """
    X_aug, Y_aug, _ = augment_dataset_np(
        np.expand_dims(img, 0),
        np.expand_dims(mask, 0)
    )
    return X_aug[0], Y_aug[0]