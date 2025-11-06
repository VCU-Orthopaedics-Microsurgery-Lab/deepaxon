'''
-------------------------------- DEEPAXON (AUGMENTED) ---------------------------------------
Train a DeepAxon++ segmentation model with CPU-based augmentation and learning rate scheduling.

Folder structure:
training/
├── images/
│   ├── image1.png # name must match mask exactly
│   ├── image2.png
├── masks/
│   ├── image1.png
│   ├── image2.png

The script will:
1. Crop images (.png or .tif) to multiples of patch size
2. Split images and masks into 256x256 patches
3. Load patches into memory
4. Apply lightweight CPU augmentations
5. Train a UNet++ model
6. Save the trained model
'''

# ------------------------------ Standard Libraries ----------------------------------- #
import os                              
import random                            

# ------------------------------ GPU Selection -------------------------------------- #
# Ask the user whether to use GPU acceleration (if available)
use_gpu = input("Use GPU acceleration if available? [y/N]: ").strip().lower()

if use_gpu not in ["y", "yes"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Suppress CUDA warnings, keep critical error messages
    print("Running DeepAxon on CPU only.")
else:
    print("Attempting to use GPU if available...")

# ------------------------------ Third-party Libraries ------------------------------ #
import cv2                              # Computer vision library 
import numpy as np                      # Numerical operations
from PIL import Image                   # Image processing
from patchify import patchify           # Split images into smaller patches

# ------------------------------ Local Imports --------------------------------------- #
from keras.utils import normalize, to_categorical       # Normalizing and categorizing dataset
from sklearn.model_selection import train_test_split    # Train/test split
from model import deepaxon_plusplus_model               # UNET++ model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from resize import resize_img                           # Custom function: resize images to a standard size

# ------------------------------ Utility Functions ----------------------------------- #
def count_files(folder_path):
    '''
    Count non-hidden files in a folder.
    '''
    return len([f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')])
    
# ------------------------------ Dataset Loader -------------------------------------- #
def load_training_dataset(images_dir, masks_dir, img_ext=".tif", mask_ext=".png"):
    '''
    Loads image and mask file paths for training, ensuring only valid files and 1:1 image-mask correspondence.
    '''
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
              if f.lower().endswith(img_ext.lower()) and os.path.isfile(os.path.join(images_dir, f))]
    
    masks = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
             if f.lower().endswith(mask_ext.lower()) and os.path.isfile(os.path.join(masks_dir, f))]

    images.sort()
    masks.sort()

    image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in images]
    mask_basenames = [os.path.splitext(os.path.basename(f))[0] for f in masks]

    matched_images, matched_masks = [], []
    for img_name, img_path in zip(image_basenames, images):
        if img_name in mask_basenames:
            matched_images.append(img_path)
            matched_masks.append(os.path.join(masks_dir, img_name + mask_ext))
        else:
            print(f"Warning: No mask found for image {img_path}, skipping.")

    print(f"Total valid image-mask pairs: {len(matched_images)}")
    return matched_images, matched_masks

# ------------------------------ Image Processing ----------------------------------------- #
def crop_center(image_path, patch_size=256):
    '''
    Crop image to nearest multiple of patch_size (256), centered
    
    :param image_path: str
        Path to a single image to be cropped.

    :returns: PIL.Image.Image
        Centered, cropped image as a PIL Image object.
    '''
    # Load given image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    # Determine new dimensions
    new_height = (height // patch_size) * patch_size
    new_width = (width // patch_size) * patch_size

    # Compute top-left coordinates for center cropping
    top = (height - new_height) // 2
    left = (width - new_width) // 2

    cropped = image[top:top+new_height, left:left+new_width]
    return Image.fromarray(cropped)

def patch(image_path, patch_size=256):
    '''
    Split single image into non-overlapping patches and save to 'patches' folder
    
    :param image_path: str
        Path to the image to patch.
        
    :param patch_size: int, optional
        Size of each square patch; default is 256.

    :returns: None
    '''
    # Extract information about the image name and path
    image_name = os.path.splitext(os.path.basename(image_path))[0] # Get the name of the image. eg. 'train/images/cropped/image1.png' --> 'image1'
    image_root = os.path.dirname(image_path)                       # Get the path to the directory the image is in. eg. 'train/images/cropped/image1.png' --> 'train/image/cropped/'
    
    # Make a subfolder named 'patches' in the root folder of the image
    patch_dir = os.path.join(image_root, 'patches')
    os.makedirs(patch_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Converts the image to black-and-white but does not save the image anywhere. just used for processing
    print(image_path)
    patches_img = patchify(image, (patch_size, patch_size), step=patch_size) # Using patchify to make patches

    # Iterate safely over patches
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            patch_array = patches_img[i, j]
            # If patchify returns shape (1,1,patch_size,patch_size), flatten it
            if patch_array.ndim == 3 and patch_array.shape[0] == 1 and patch_array.shape[1] == 1:
                single_patch = patch_array[0,0]
            else:
                single_patch = patch_array
            patch_path = os.path.join(patch_dir, f"{image_name}_{i}{j}.png")
            cv2.imwrite(patch_path, single_patch)

def batch_patch(images, masks, patch_size=256):
    '''
    Crop and patch lists of images and masks
    '''
    for path in images + masks:
        # Create cropped folder
        cropped_dir = os.path.join(os.path.dirname(path), "cropped")
        os.makedirs(cropped_dir, exist_ok=True)
        # Crop
        cropped_img = crop_center(path, patch_size)
        cropped_path = os.path.join(cropped_dir, os.path.basename(path))
        cropped_img.save(cropped_path)
        # Patch
        patch(cropped_path, patch_size)
        
def get_images(patch_path):
    '''
    Return all images in folder as numpy array
    
    :param patch_path: A path (string or object) 
    
    :returns: Numpy Array; all images in the patches folder
    '''
    #go to every list in the patch folder and append it to the list
    patch_files = [os.path.join(patch_path, f) for f in os.listdir(patch_path)
                   if os.path.isfile(os.path.join(patch_path, f)) and not f.startswith('.')]
    patch_files.sort()
    return np.array([resize_img(f) for f in patch_files])     # resize 2048 resolution images to 1024 for training

def base_label(train_masks):
    '''
    If the manual masks are made with colors instead of 0,1,2 this function will convert it into 0,1,2.
    
    :param train_masks: Numpy Array; training mask images
    
    :returns: Numpy Array; training mask images with values being 0,1,2 instead of whatever colors it was
    '''
    
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    return np.vectorize(lambda x: pixel_to_class.get(x,0))(train_masks)


# ------------------------------ Augmentation Functions -------------------------------- #
def augment_dataset_np(X, Y):
    """
    Apply lightweight augmentations to a batch of images (X) and masks (Y)
    CPU-based using numpy and OpenCV:
    - Horizontal / vertical flips
    - Small rotations
    - Brightness / contrast adjustments
    - Gamma adjustment
    - Additive Gaussian noise
    """
    X_aug, Y_aug = [], []
    for img, mask in zip(X, Y):
        img_aug, mask_aug = img.copy(), mask.copy()

        # Random horizontal flip
        if random.random() < 0.5:
            img_aug = np.flip(img_aug, axis=1)
            mask_aug = np.flip(mask_aug, axis=1)

        # Random vertical flip
        if random.random() < 0.5:
            img_aug = np.flip(img_aug, axis=0)
            mask_aug = np.flip(mask_aug, axis=0)

        # Small rotation (-10 to +10 degrees)
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((img_aug.shape[1]/2, img_aug.shape[0]/2), angle, 1)
        img_aug = cv2.warpAffine(img_aug, M, (img_aug.shape[1], img_aug.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_aug = cv2.warpAffine(mask_aug, M, (mask_aug.shape[1], mask_aug.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Brightness / contrast
        alpha = random.uniform(0.9, 1.1)  # contrast
        beta = random.uniform(-10, 10)    # brightness
        img_aug = np.clip(img_aug * alpha + beta, 0, 255).astype(np.uint8)

        # Gamma correction
        gamma = random.uniform(0.95, 1.05)
        img_aug = np.clip(255.0 * (img_aug / 255.0) ** gamma, 0, 255).astype(np.uint8)

        # Add Gaussian noise
        noise = np.random.normal(0, 2, img_aug.shape)
        img_aug = np.clip(img_aug + noise, 0, 255).astype(np.uint8)

        X_aug.append(img_aug)
        Y_aug.append(mask_aug)

    return np.array(X_aug), np.array(Y_aug)

# ------------------------------ Model Training -------------------------------------------------- #
def train_model(training_dir, model_path, model_name, batch_size=16, epochs=200, test_fraction=0.3, img_ext=".tif", mask_ext=".png"):
    '''
    Train and save DeepAxon++ model with learning rate scheduler and optional early stopping.
    
    :param training_dir: Folder containing 'images/' and 'masks/'.
    :param model_path: Folder to save trained model.
    :param model_name: Filename for saved model.
    :param batch_size: Batch size for training.
    :param epochs: Maximum number of epochs to train.
    :param test_fraction: Fraction of dataset to reserve for testing.
    :param img_ext: Image file extension (default ".tif").
    :param mask_ext: Mask file extension (default ".png").
    
    :returns: Trained Keras model
    '''
    images_dir = os.path.join(training_dir, "images")
    masks_dir = os.path.join(training_dir, "masks")

    # Load valid image-mask pairs
    images, masks = load_training_dataset(images_dir, masks_dir, img_ext, mask_ext)

    # Crop and patch all images & masks
    batch_patch(images, masks, 512) # double patch size from 256 for 100x images

    # Paths to patch folders
    image_patch_path = os.path.join(images_dir, "cropped", "patches")
    mask_patch_path  = os.path.join(masks_dir, "cropped", "patches")

    # Load patches into memory
    X = normalize(get_images(image_patch_path), axis=1) # [M] This may normalize differently than in segment.py
    Y = base_label(get_images(mask_patch_path))
    X = np.expand_dims(X, axis=3)   # 4D: (n_samples, H, W, 1)
    Y = np.expand_dims(Y, axis=3)

    # Dataset summary
    print("\nDataset summary:")
    print(f"  Original images: {count_files(images_dir)}")
    print(f"  Original masks:  {count_files(masks_dir)}")
    print(f"  Image patches:   {X.shape[0]}")
    print(f"  Mask patches:    {Y.shape[0]}\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_fraction, random_state=0)
    
    # Apply augmentation to training set
    # Soft coded for now, once finalized can integrate into model.py with tensorflow
    print("[INFO] Applying CPU-based augmentations...")
    X_train_aug, y_train_aug = augment_dataset_np(X_train, y_train) # augment training set 
    # Combine original and augmented images to expand dataset (OPTIONAL)
    X_train_combined = np.concatenate([X_train, X_train_aug])
    y_train_combined = np.concatenate([y_train, y_train_aug])
    
    # Convert masks to one-hot encoding
    y_train_cat = to_categorical(y_train_combined, num_classes=3)
    y_test_cat  = to_categorical(y_test, num_classes=3)

    # Build model
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1:4]
    model = deepaxon_plusplus_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------- Callbacks ------------------------------------- #
    # Reduce learning rate when validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',     # Track validation loss
        factor=0.5,             # Reduce LR by half when plateau occurs
        patience=30,            # Wait 30 epochs without improvement before reducing LR
        verbose=1, 
        min_lr=1e-6             # Don't go below this learning rate
    )

    # Early stopping to prevent overfitting and restore best weights
    early_stop = EarlyStopping(
        monitor='val_loss',     # Stop when validation loss stops improving
        patience=60,            # Wait 60 epochs without improvement before stopping
        restore_best_weights=True, 
        verbose=1
    )
    
    # Train model
    print("[INFO] Starting training...")
    history = model.fit(
        X_train_combined, y_train_cat,
        validation_data=(X_test, y_test_cat),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=False,
        callbacks=[lr_scheduler, early_stop]  # use both callbacks
    )

    # Save model
    os.makedirs(model_path, exist_ok=True)
    full_model_path = os.path.join(model_path, model_name + ".keras")
    model.save(full_model_path)

    # Print summary
    print(f"\nModel saved at: {full_model_path}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    return model