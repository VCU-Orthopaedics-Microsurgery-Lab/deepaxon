'''
-------------------------------- DEEPAXON --------------------------------
obtain segmented image file where the meylin is a middle grey and the axons are white
'''
# ------------------------------ Standard Libraries ---------------------------------- #
import os
import time
from datetime import datetime
import csv
from PIL import Image                   # Image processing
import numpy as np                      # Numerical operations

# ------------------------------ Third-Party Libraries ------------------------------- # 
import cv2                              # Computer vision library 
import tensorflow as tf
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
    i_max, j_max = shape[0]-1, shape[1]-1
    if i == 0 and j == 0: return 0
    if i == 0 and j == j_max: return 2
    if i == i_max and j == 0: return 6
    if i == i_max and j == j_max: return 8
    if i == 0: return 1
    if i == i_max: return 7
    if j == 0: return 3
    if j == j_max: return 5
    return 4

def hann_fn(x, patch_size):
    '''
    1D Hann window function scaled to patch size.
    '''
    return (1 - np.cos(2 * np.pi * x / (patch_size - 1))) / 2

def hann_window(pos, patch_size):
    '''
     Generate 2D Hann window for a given patch position and size.
    '''
    i, j = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')
    center = patch_size // 2
    cond1 = (i <= center) & (j <= center)
    cond2 = (i > center) & (j <= center)
    cond3 = (i <= center) & (j > center)
    cond4 = ~cond1 & ~cond2 & ~cond3
    
    scaler = np.zeros((patch_size, patch_size), dtype=float)
    
    # Hann weighting for each of the 9 possible patch positions
    if pos == 0:
        scaler[cond1] = 1
        scaler[cond2] = hann_fn(i[cond2], patch_size)
        scaler[cond3] = hann_fn(j[cond3], patch_size)
        scaler[cond4] = hann_fn(i[cond4], patch_size) * hann_fn(j[cond4], patch_size)
    elif pos == 1:
        scaler[cond1] = hann_fn(j[cond1], patch_size)
        scaler[cond2] = hann_fn(i[cond2], patch_size) * hann_fn(j[cond2], patch_size)
        scaler[cond3] = hann_fn(j[cond3], patch_size)
        scaler[cond4] = hann_fn(i[cond4], patch_size) * hann_fn(j[cond4], patch_size)
    elif pos == 2:
        scaler[cond1] = hann_fn(j[cond1], patch_size)
        scaler[cond2] = hann_fn(i[cond2], patch_size) * hann_fn(j[cond2], patch_size)
        scaler[cond3] = 1
        scaler[cond4] = hann_fn(i[cond4], patch_size)
    elif pos == 3:
        scaler[cond1] = hann_fn(i[cond1], patch_size)
        scaler[cond2] = hann_fn(i[cond2], patch_size)
        scaler[cond3] = hann_fn(i[cond3], patch_size) * hann_fn(j[cond3], patch_size)
        scaler[cond4] = hann_fn(i[cond4], patch_size) * hann_fn(j[cond4], patch_size)
    elif pos == 4:
        scaler[cond1] = hann_fn(i[cond1], patch_size) * hann_fn(j[cond1], patch_size)
        scaler[cond2] = hann_fn(i[cond2], patch_size) * hann_fn(j[cond2], patch_size)
        scaler[cond3] = hann_fn(i[cond3], patch_size) * hann_fn(j[cond3], patch_size)
        scaler[cond4] = hann_fn(i[cond4], patch_size) * hann_fn(j[cond4], patch_size)
    elif pos == 5:
        scaler[cond1] = hann_fn(i[cond1], patch_size) * hann_fn(j[cond1], patch_size)
        scaler[cond2] = hann_fn(i[cond2], patch_size) * hann_fn(j[cond2], patch_size)
        scaler[cond3] = hann_fn(i[cond3], patch_size)
        scaler[cond4] = hann_fn(i[cond4], patch_size)
    elif pos == 6:
        scaler[cond1] = hann_fn(i[cond1], patch_size)
        scaler[cond2] = 1
        scaler[cond3] = hann_fn(i[cond3], patch_size) * hann_fn(j[cond3], patch_size)
        scaler[cond4] = hann_fn(j[cond4], patch_size)
    elif pos == 7:
        scaler[cond1] = hann_fn(i[cond1], patch_size) * hann_fn(j[cond1], patch_size)
        scaler[cond2] = hann_fn(j[cond2], patch_size)
        scaler[cond3] = hann_fn(i[cond3], patch_size) * hann_fn(j[cond3], patch_size)
        scaler[cond4] = hann_fn(j[cond4], patch_size)
    elif pos == 8:
        scaler[cond1] = hann_fn(i[cond1], patch_size) * hann_fn(j[cond1], patch_size)
        scaler[cond2] = hann_fn(j[cond2], patch_size)
        scaler[cond3] = hann_fn(i[cond3], patch_size)
        scaler[cond4] = 1
        
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
            hann_matrix = hann_window(patch_pos, patch_size)
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
def segment_dir(dir_path, model, output_path, patch_size=256):
    '''
    Apply segmentation to all images in a folder using the trained UNet++ model.
    Loads the model once to avoid reloading for every image.
    Per Image timing (print + csv saved)
    '''
    valid_exts = ('.tif', '.tiff', '.png')
    times = {}
    folder_start = time.time() #folder-level timer
    
    for img_name in os.listdir(dir_path):
        if img_name != "Thumbs.db" and img_name.lower().endswith(valid_exts):
            img_path = os.path.join(dir_path, img_name)
            start_time = time.time()
            
            try:
                segment(img_path, model, output_path, patch_size)
                
            except Exception as e:
                print(f"Error processing {img_name}:{e}")
                continue
            
            elapsed = time.time() - start_time
            times[img_name] = elapsed
            
            print(f"Segmented {img_name} in {elapsed:.2f} s")
            
    total_time = sum(times.values())
    avg_time = total_time / len(times) if times else 0.0
    folder_end = time.time()
        
    print(f"\nFolder '{dir_path}' segmentation complete ✅")
    print(f"Total segmentation time (images only): {total_time:.2f} s")
    print(f"Average time per image: {avg_time:.2f} s")
    print(f"Total folder runtime (including overhead): {folder_end - folder_start:.2f} s")
            
    # Save CSV in output folder
    date_str = datetime.now().strftime("%Y%m%d")  # e.g., 20260212
    csv_path = os.path.join(output_path, f"segmentation_times_{date_str}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Time_s"])
        for img_name, t in times.items():
            writer.writerow([img_name, t])
                
    print(f"Segmentation times saved to {csv_path}")
    return times