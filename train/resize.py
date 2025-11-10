'''
-------------------------------- DEEPAXON --------------------------------
resize grayscale images for DeepAxon processing; only resizes specific large images
'''
# ------------------------------ Standard Library ----------------------------------- #
import os

# ------------------------------ Third-Party Libraries ------------------------------ #
import cv2                              # Computer vision library 

# ------------------------------ Image Resizing Function ----------------------------- #
def resize_img(img_path):
    '''
    Load a grayscale image and optionally resize it if it matches a specific resolution.

    Args:
        img_path (str): Path to the input image.

    Returns:
        np.ndarray: Grayscale image, resized if original dimensions were (2048, 2880).
    '''
    img = cv2.imread(img_path, 0)               # Load image in grayscale
    if img is None:
        raise ValueError(f"Error: could not read image at {img_path}")

    # Only resize if image matches this exact dimension
    if img.shape == (2048, 2880):               
        return cv2.resize(img, (1440, 1024))    # cv2.resize expects (width, height)
    
    # Warn if image does not conform
    if img.shape != (1024, 1440):
        print(f"Warning: Image {os.path.basename(img_path)} has unexpected dimensions {img.shape}.")
        
    return img                                  # Otherwise, return original