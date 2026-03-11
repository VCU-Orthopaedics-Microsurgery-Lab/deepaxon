"""
-------------------------------- DEEPAXON --------------------------------
resize grayscale images for DeepAxon processing; only resizes specific large images
"""
# ------------------------------ Standard Library ------------------------------------ #
import os

# ------------------------------ Third-party Libraries ------------------------------- #
import cv2                              # Computer vision (OpenCV) 
import numpy as np                      # Numerical operations

# ------------------------------ Image Resizing Function ----------------------------- #
def resize_img(img_path, is_mask=False):
    """
    Load a grayscale image (or mask) and resizes it if it matches a specific resolution.
    For masks, use nearest-neighbor interpolation and force class values to {0,127,255}.

    Args:
        img_path (str): Path to the input image.
        is_mask (bool): If True, treat the image as a label mask.

    Returns:
        np.ndarray: Grayscale image, resized if original dimensions were (2048, 2880).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                                    # Load image in grayscale
    if img is None:
        raise ValueError(f"Error: could not read image at {img_path}")

    # Only resize if image matches this exact dimension
    if img.shape == (2048, 2880):
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR # nearest neighbor to preserve mask integrity               
        img = cv2.resize(img, (1440, 1024), interpolation=interp)  # cv2.resize expects (width, height)
    
    # Warn if image does not conform
    elif img.shape != (1024, 1440):
        print(f"Warning: Image {os.path.basename(img_path)} has unexpected dimensions {img.shape}.")
        
    # If it's a mask, ensure exact class values (0, 127, 255)
    if is_mask:
        # Fix tiny deviations (e.g., 126–129)
        img = np.where((img >= 126) & (img <= 129), 127, img)
        img = np.where(img > 200, 255, img)
        img = np.where(img < 50, 0, img)

    return img