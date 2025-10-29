'''
-------------------------------- DEEPAXON --------------------------------
obtain segmented image file where the meylin is a middle grey and the axons are white
'''
# ------------------------------ Standard Libraries ---------------------------------- #
import os
from PIL import Image                   # Image processing

# ------------------------------ Third-Party Libraries ------------------------------- # 
import cv2                              # Computer vision library 
import numpy as np                      # Numerical operations
from keras.models import load_model     # Load trained UNet++ model
from keras.utils import normalize       # Normalize image pixel values
from patchify import patchify           # Splits large images into smaller overlapping patches

# ------------------------------- Local Imports -------------------------------------- #
from resize import resize_img           # Custom function: resize images to a standard size

# ------------------------------- Patch Utilities ------------------------------------- #
def get_pos(shape, i,j):
    '''
    Determine the relative position of a patch within the full image.
    Returns an integer (0-8) indicating patch location (corner, edge, or center).
    '''
    i_max = shape[0]-1
    j_max = shape[1]-1
    if i == 0 and j == 0:
        pos = 0
    elif i == 0 and j == j_max:
        pos = 2
    elif i == i_max and j == 0:
        pos = 6
    elif i == i_max and j == j_max:
        pos = 8
    elif i == 0:
        pos = 1
    elif i == i_max:
        pos = 7
    elif j == 0:
        pos = 3
    elif j == j_max:
        pos = 5
    else:
        pos = 4
    return pos

def hann_fn(x):
    '''
    Hann window function for smoothing patch edges.
    '''
    return (1 - np.cos(2 * np.pi * x / 255))/2

def hann_window(pos):
    '''
    Generate a 2D Hann window matrix based on patch position.
    Helps smoothly blend overlapping patches.
    '''
    i, j = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
    condition1 = (i <= 128) & (j <= 128)
    condition2 = (i > 128) & (j < 128)
    condition3 = (i < 128) & (j > 128)
    condition4 = ~condition1 & ~condition2 & ~condition3
    
    scaler = np.zeros((256,256), dtype=float)
    # Hann weighting for each of the 9 possible patch positions
    if pos == 0:
        scaler[condition1] = 1
        scaler[condition2] = hann_fn(i[condition2])
        scaler[condition3] = hann_fn(j[condition3])
        scaler[condition4] = hann_fn(i[condition4]) * hann_fn(j[condition4])
    elif pos == 1:
        scaler[condition1] = hann_fn(j[condition1])
        scaler[condition2] = hann_fn(i[condition2]) * hann_fn(j[condition2])
        scaler[condition3] = hann_fn(j[condition3])
        scaler[condition4] = hann_fn(i[condition4]) * hann_fn(j[condition4])
    elif pos == 2:
        scaler[condition1] = hann_fn(j[condition1])
        scaler[condition2] = hann_fn(i[condition2]) * hann_fn(j[condition2])
        scaler[condition3] = 1
        scaler[condition4] = hann_fn(i[condition4])
    elif pos == 3:
        scaler[condition1] = hann_fn(i[condition1])
        scaler[condition2] = hann_fn(i[condition2])
        scaler[condition3] = hann_fn(i[condition3]) * hann_fn(j[condition3])
        scaler[condition4] = hann_fn(i[condition4]) * hann_fn(j[condition4])
    elif pos == 4:
        scaler[condition1] = hann_fn(i[condition1]) * hann_fn(j[condition1])
        scaler[condition2] = hann_fn(i[condition2]) * hann_fn(j[condition2])
        scaler[condition3] = hann_fn(i[condition3]) * hann_fn(j[condition3])
        scaler[condition4] = hann_fn(i[condition4]) * hann_fn(j[condition4])
    elif pos == 5:
        scaler[condition1] = hann_fn(i[condition1]) * hann_fn(j[condition1])
        scaler[condition2] = hann_fn(i[condition2]) * hann_fn(j[condition2])
        scaler[condition3] = hann_fn(i[condition3])
        scaler[condition4] = hann_fn(i[condition4])
    elif pos == 6:
        scaler[condition1] = hann_fn(i[condition1])
        scaler[condition2] = 1
        scaler[condition3] = hann_fn(i[condition3]) * hann_fn(j[condition3])
        scaler[condition4] = hann_fn(j[condition4])
    elif pos == 7:
        scaler[condition1] = hann_fn(i[condition1]) * hann_fn(j[condition1])
        scaler[condition2] = hann_fn(j[condition2])
        scaler[condition3] = hann_fn(i[condition3]) * hann_fn(j[condition3])
        scaler[condition4] = hann_fn(j[condition4])
    elif pos == 8:
        scaler[condition1] = hann_fn(i[condition1]) * hann_fn(j[condition1])
        scaler[condition2] = hann_fn(j[condition2])
        scaler[condition3] = hann_fn(i[condition3])
        scaler[condition4] = 1
        
    return scaler

# ------------------------------ Visualization ----------------------------------- #
def recolor(img):
    '''
    Map integer labels (0=background, 1=myelin, 2=axon) to RGB colors for visualization.
    '''
    colors = {
        0: (0, 0, 0),           # background
        1: (128, 128, 128),     # myelin
        2: (255, 255, 255),     # axon
    }
    img_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        img_color[img == value, :] = color 
    return img_color

# ------------------------------ Image Segmentation ------------------------------ #
# Main segmentation function for a single image
def segment(img_path, model, output_path, patch_size=256):
    '''
    Segment a single image using the trained UNet++ model.
    Applies patch-based prediction with Hann window blending.
    '''
    img = resize_img(img_path)             # Resize image to standard dimensions

    
    SIZE_X = img.shape[1] // patch_size * patch_size
    SIZE_Y = img.shape[0] // patch_size * patch_size
    
    img = Image.fromarray(img)
    img = img.crop((0,0,SIZE_X, SIZE_Y))    # Ensure divisible by patch size
    img = np.array(img)
    
    patches = patchify(img, (patch_size, patch_size), patch_size//2) # 50% overlap
    
    pred_img = np.zeros(img.shape)  # Placeholder for reconstructed prediction
    
    # Loop through patches and predict
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i,j,:,:]
            patch = normalize(patch)
            patch = np.expand_dims(patch, axis=(0,3))       # Add batch and channel dims
            pred = model.predict(patch)                     # UNet++ prediction
            pred = np.argmax(pred, axis=3)[0,:,:]           # Remove batch dim
            
            patch_pos = get_pos(patches.shape, i, j)
            hann_matrix = hann_window(patch_pos)
            adj_pred = pred * hann_matrix                   # Apply Hann weighting
            
            i_start = i*patch_size//2
            i_end = i_start+patch_size
            j_start = j*patch_size//2
            j_end = j_start+patch_size
            pred_img[i_start:i_end, j_start:j_end] += adj_pred
            
    pred_img = np.round(pred_img).astype(int)
    pred_img = recolor(pred_img)
    
    pred_img = Image.fromarray(pred_img)
    img_name = os.path.basename(img_path)
    pred_path = os.path.join(output_path,img_name.split('.')[0] + "_seg." + img_name.split('.')[1])
    pred_img.save(pred_path)
    
    return pred_path

# ------------------------------ Directory Segmentation ------------------------------ #
def segment_dir(dir_path, model_path, output_path):
    '''
    Apply segmentation to all images in a folder using the trained UNet++ model.
    Loads the model once to avoid reloading for every image.
    '''
    model = load_model(model_path)      # Load trained UNet++ model
    for img_name in os.listdir(dir_path):
        if img_name != "Thumbs.db":     # Skip OS artifact files
            img_path = os.path.join(dir_path,img_name)
            segment(img_path, model_path, output_path)