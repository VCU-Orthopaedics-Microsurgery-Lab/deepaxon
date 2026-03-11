#!/usr/bin/env python3
"""
DeepAxon++ Quick Visual Debug Tool
---------------------------------
Loads a few image/mask pairs and, if a trained model is available,
runs predictions and visualizes them side-by-side.
Helps verify data alignment, normalization, and segmentation quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------- USER SETTINGS -------------------- #
# Adjust these paths for your local setup
train_dir   = r"T:\Orthopaedics\Lab Imaging Data\mmazur\DA_Model_Training\100X\100X Model Training\Selected Images\trainingv3a"                    # directory containing X.npy and Y.npy
model_path  = r"C:\Users\mazurm\deepaxon\models\p_100x_auglogtest1.keras"  # path to trained model (optional)
num_samples = 5  # number of samples to visualize

# -------------------------------------------------------- #

def load_data():
    image_dir = r"T:\Orthopaedics\Lab Imaging Data\mmazur\DA_Model_Training\100X\100X Model Training\Selected Images\trainingv3a\images\cropped\patches"
    mask_dir  = r"T:\Orthopaedics\Lab Imaging Data\mmazur\DA_Model_Training\100X\100X Model Training\Selected Images\trainingv3a\masks\cropped\patches"

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    mask_paths  = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not image_paths or not mask_paths:
        raise FileNotFoundError("No images found in the specified directories.")

    X, Y = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = img_to_array(load_img(img_path, color_mode="rgb"))
        mask = img_to_array(load_img(mask_path, color_mode="grayscale"))
        X.append(img)
        Y.append(mask)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def class_balance(Y):
    """Compute and print pixel-level class balance."""
    print("\n📊 Pixel Class Distribution:")
    unique, counts = np.unique(Y, return_counts=True)
    total = np.sum(counts)
    for u, c in zip(unique, counts):
        print(f" - Class {int(u):>2}: {c:,} pixels ({c / total * 100:.2f}%)")
    print("-" * 50)

def visualize_alignment(X, Y, idx=0):
    """Visualize image-mask alignment for a given index."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X[idx])
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(Y[idx], axis=-1) if Y.ndim == 4 else Y[idx], cmap='jet')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(X[idx])
    plt.imshow(np.argmax(Y[idx], axis=-1) if Y.ndim == 4 else Y[idx],
               cmap='jet', alpha=0.5)
    plt.title("Overlay (Check Alignment)")
    plt.axis("off")
    plt.show()

def predict_sample(model, X, Y, idx=0):
    """Run model prediction and compare visually."""
    sample = X[idx:idx+1].astype("float32") / 255.0
    pred = model.predict(sample)
    pred_mask = np.argmax(pred[0], axis=-1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X[idx])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(Y[idx], axis=-1) if Y.ndim == 4 else Y[idx], cmap='jet')
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='jet')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.suptitle(f"Sample {idx} — Model Segmentation Output", fontsize=14)
    plt.show()

def main():
    X, Y = load_data()
    class_balance(Y)

    # Pick a few random samples
    sample_indices = np.random.choice(X.shape[0], num_samples, replace=False)
    for i, idx in enumerate(sample_indices, start=1):
        print(f"\n🧩 Visualizing sample {i}/{num_samples} (index {idx})")
        visualize_alignment(X, Y, idx)

    # If a model exists, run predictions
    if os.path.exists(model_path):
        print(f"\n🤖 Loading model from: {model_path}")
        model = load_model(model_path, compile=False)
        test_idx = np.random.choice(X.shape[0])
        print(f"Running prediction on random sample #{test_idx}")
        predict_sample(model, X, Y, test_idx)
    else:
        print("\n⚠️ No model found at given path. Skipping prediction test.")

if __name__ == "__main__":
    main()
