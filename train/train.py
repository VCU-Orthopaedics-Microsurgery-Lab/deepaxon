'''
-------------------------------- DEEPAXON (AUGMENTED) ---------------------------------------
Train a DeepAxon++ segmentation model with CPU-based augmentation and learning rate scheduling.

Folder structure:
training/
├── images/
│   ├── image1.png      # name must match mask exactly
│   ├── image2.png
├── masks/
│   ├── image1.png
│   ├── image2.png

The script will:
1) Resize large images to standard size (only original images)
2) Crop images (.png or .tif) to multiples of patch size
3) Split images and masks into patches
4) Load patches into memory
5) Apply lightweight CPU augmentations (patches only; masks transformed with geometry only)
6) Train a UNet++ model with LR scheduler + early stopping
7) Save the trained model
8) Save a human-readable training log alongside the model
'''

# ------------------------------ Standard Libraries ------------------------------------ #
import os                              
import random   
from datetime import datetime                         

# ------------------------------ GPU Selection ----------------------------------------- #
# Ask whether to try GPU; otherwise force CPU (works fine on CPU too).
use_gpu = input("Use GPU acceleration if available? [y/N]: ").strip().lower()

if use_gpu not in ["y", "yes"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Suppress CUDA warnings, keep critical error messages
    print("Running DeepAxon on CPU only.")
else:
    print("Attempting to use GPU if available...")

# ------------------------------ Third-party Libraries --------------------------------- #
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import cv2                              # Computer vision library 
import numpy as np                      # Numerical operations
from PIL import Image                   # Image processing
from patchify import patchify           # Split images into smaller patches
import tensorflow as tf

# ------------------------------ Keras / Sklearn Imports ------------------------------- #
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from sklearn.model_selection import train_test_split

# ------------------------------ Local Imports ----------------------------------------- #
from model import deepaxon_plusplus_model               # UNET++ model
from resize import resize_img                           # Custom function: resize images to a standard size


# ====================================================================================== #
#                                   UTILITY HELPERS                                      #
# ====================================================================================== #

def count_files(folder_path):
    '''
    Count non-hidden files in a folder.
    '''
    return len([f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')])
    
# ------------------------------ Dataset Loader -------------------------------------- #
def load_training_dataset(images_dir, masks_dir, img_ext=".tif", mask_ext=".png"):
    '''
    Load image/mask paths and enforce 1:1 pairing by basename.
    Only pairs that exist in both folders are kept.
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

    print(f"[CHECK] Found {len(matched_images)} valid image-mask pairs before preprocessing.")
    return matched_images, matched_masks


# ====================================================================================== #
#                                  IMAGE PROCESSING                                      #
# ====================================================================================== #

console = Console()

def crop_center_array(img_array, patch_size=256):
    '''
    Center-crop a numpy array so BOTH dimensions are exact multiples of patch_size.

    Example:
        If resized image is 1024×1440 and patch_size=256:
            New height = 1024 (already multiple)
            New width  = 1024 (1440 cropped to 1024)
    '''
    height, width = img_array.shape[:2]
    new_height = (height // patch_size) * patch_size
    new_width  = (width  // patch_size) * patch_size
    top = (height - new_height) // 2
    left = (width  - new_width)  // 2
    cropped = img_array[top:top+new_height, left:left+new_width]
    return cropped

def patch_array_and_save(img_array, save_dir, image_name, patch_size=256):
    os.makedirs(save_dir, exist_ok=True)
    patches = patchify(img_array, (patch_size, patch_size), step=patch_size)
    patch_count = 0

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_array = patches[i, j]
            if patch_array.ndim == 3 and patch_array.shape[0] == 1:
                patch_array = patch_array[0]
            patch_path = os.path.join(save_dir, f"{image_name}_{i}{j}.png")
            cv2.imwrite(patch_path, patch_array)
            patch_count += 1

    num_channels = img_array.shape[2] if img_array.ndim == 3 else 1
    return patch_count, num_channels
    
def process_single_image(path, patch_size=256):
    image_name = os.path.splitext(os.path.basename(path))[0]
    parent_dir = os.path.dirname(path)

    original_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    orig_shape = original_img.shape
    orig_channels = 1 if len(orig_shape) == 2 else orig_shape[2]

    resized_img = resize_img(path)
    resized_shape = resized_img.shape
    resized_channels = 1 if len(resized_shape) == 2 else resized_shape[2]

    # Flatten single-channel 3D arrays (HxWx1)
    if resized_img.ndim == 3 and resized_img.shape[2] == 1:
        resized_img = resized_img[:, :, 0]

    cropped_img = crop_center_array(resized_img, patch_size=patch_size)
    cropped_shape = cropped_img.shape
    cropped_channels = 1 if len(cropped_shape) == 2 else cropped_shape[2]

    cropped_dir = os.path.join(parent_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    cropped_path = os.path.join(cropped_dir, os.path.basename(path))
    cv2.imwrite(cropped_path, cropped_img)

    patch_dir = os.path.join(cropped_dir, "patches")
    patch_count, num_channels = patch_array_and_save(cropped_img, patch_dir, image_name, patch_size)

    # Render all in one Rich panel
    console.print(
        Panel.fit(
            f"[bold cyan]{image_name}[/bold cyan]\n"
            f"[white]Original:[/white] {orig_shape} → [yellow]{orig_channels} ch[/yellow]\n"
            f"[white]Resized:[/white]  {resized_shape} → [yellow]{resized_channels} ch[/yellow]\n"
            f"[white]Cropped:[/white]  {cropped_shape} → [yellow]{cropped_channels} ch[/yellow]\n"
            f"[white]Patches:[/white]  {patch_count} total → [yellow]{num_channels} ch each[/yellow]\n"
            f"[white]Saved:[/white]    [green]{patch_dir}[/green]",
            title=f"[PROCESS] {image_name}",
            border_style="cyan",
        )
    )
    

def batch_patch(images, masks, patch_size=256):
    '''
    Apply full processing to ALL images AND masks.
    Includes section headers for readability
    '''
    console.rule("[bold yellow]IMAGE PROCESSING START[/bold yellow]")
    for p in images:
        process_single_image(p, patch_size=patch_size)
        
    console.rule("[bold yellow]MASK PROCESSING START[/bold yellow]")
    for p in masks:
        process_single_image(p, patch_size=patch_size)
    
def get_images(patch_path):
    '''
    Read all image files under 'patch_path' (sorted) and return as a numpy array.
    
    :param patch_path: A path (string or object) 
    
    :returns: Numpy Array; all images in the patches folder
    '''
    #go to every list in the patch folder and append it to the list
    files = [os.path.join(patch_path, f) for f in os.listdir(patch_path)
                   if os.path.isfile(os.path.join(patch_path, f)) and not f.startswith('.')]
    files.sort()
    return np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])

def base_label(train_masks):
    '''
    Convert grayscale mask pixels into class IDs (0,1,2).
    Any unknown value defaults to background (0).
    
    :param train_masks: Numpy Array; training mask images
    
    :returns: Numpy Array; training mask images with values being 0,1,2 instead of whatever colors it was
    '''
    # Ensure integer mask values
    train_masks = np.round(train_masks).astype(np.uint8)
    
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    return np.vectorize(lambda x: pixel_to_class.get(x,0))(train_masks)


# ===================================================================================== #
#                                   AUGMENTATION                                        #
# ===================================================================================== #

def augment_dataset_np(X,
                       Y,
                       aug_prob=0.3,
                       p_flip_h=0.5,
                       p_flip_v=0.5,
                       p_rotate=0.5,
                       p_brightness=0.3,
                       p_gamma=0.3,
                       p_noise=0.2):
    '''
   Perform lightweight, on-the-fly CPU augmentations on (X, Y) batches.

    Augmentation strategy:
    - A global gatekeeper (`aug_prob`) determines whether any augmentation is applied to a given patch.
    - If augmentation is triggered, each augmentation type is applied independently with its own probability:
        - Random horizontal and vertical flips
        - Small random rotation (-10° to +10°) 
            - Masks use nearest-neighbor interpolation
            - Images use linear interpolation
        - Small random brightness and contrast adjustments (applied to images only)
        - Small random gamma adjustment (applied to images only)
        - Small additive Gaussian noise (applied to images only)
        
    NOTE: Masks are only changed by geometry (flips/rotations); no intensity changes.
    
    :returns: X_out: output images (augmented or original)
              Y_out: corresponding masks (geometrically transformed if augmented)
              flags: boolean array; True if patch was augmented, False otherwise
    '''
    X_out, Y_out, flags = [], [], []
    
    for img, mask in zip(X, Y):
        img_out, mask_out = img.copy(), mask.copy()
        augmented = False
        
        # Decide if augmentation occurs
        if random.random() < aug_prob:

            # Random horizontal flip
            if random.random() < p_flip_h:
                img_out = np.flip(img_out, axis=1)
                mask_out = np.flip(mask_out, axis=1)
                augmented = True

            # Random vertical flip
            if random.random() < p_flip_v:
                img_out = np.flip(img_out, axis=0)
                mask_out = np.flip(mask_out, axis=0)
                augmented = True

            # Random small rotation (-10 to +10 degrees)
            if random.random() < p_rotate:
                angle = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((img_out.shape[1]/2, img_out.shape[0]/2), angle, 1)

                # Image: linear interpolation
                img_out = cv2.warpAffine(img_out, M, (img_out.shape[1], img_out.shape[0]),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # Mask: nearest-neighbor, safe cast & clip
                mask_out = mask_out.astype(np.uint8)
                mask_out = cv2.warpAffine(mask_out, M, (mask_out.shape[1], mask_out.shape[0]),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                mask_out = np.round(mask_out).astype(np.uint8)
                mask_out = np.clip(mask_out, 0, 2)
                augmented = True

            # Random brightness / contrast jitter (images only)
            if random.random() < p_brightness:
                alpha = random.uniform(0.9, 1.1)  # contrast
                beta = random.uniform(-0.1, 0.1) # brightness in 0–1 scale
                img_out = np.clip(img_out * alpha + beta, 0.0, 1.0)
                augmented = True

            # Random Gamma jitter (images only)
            if random.random() < p_gamma:
                gamma = random.uniform(0.95, 1.05)
                img_out = np.clip(img_out ** gamma, 0.0, 1.0)
                augmented = True

            # Random tiny Gaussian noise (images only)
            if random.random() < p_noise:
                noise = np.random.normal(0, 0.01, img_out.shape)  # small noise in 0–1 range
                img_out = np.clip(img_out + noise, 0.0, 1.0)
                augmented = True
            
        # If image/mask is 2D (H x W), add channel dimension
        if img_out.ndim == 2:
            img_out = np.expand_dims(img_out, axis=-1)  # shape: H x W x 1
        if mask_out.ndim == 2:
            mask_out = np.expand_dims(mask_out, axis=-1)

        # Always append, whether augmented or not
        X_out.append(img_out)
        Y_out.append(mask_out)
        flags.append(augmented)

    return np.array(X_out), np.array(Y_out), np.array(flags)


# ===================================================================================== #
#                                  EPOCH LOG CALLBACK                                   #
# ===================================================================================== #

class TrainingLogger(Callback):
    '''
    Unified logger that combines:
      - Epoch metrics (loss, accuracy, val_loss, val_accuracy, LR)
      - Augmentation stats (total_aug, avg_per_batch, min_batch, max_batch)
    
    Writes a neatly formatted table to both console and file:
    
    Epoch |    Loss |   Acc | Val_Loss | Val_Acc |    LR    | Aug(total/avg/min/max)
    -------------------------------------------------------------------------------
        1 |  1.1780 | 0.671 |  1.40035 | 0.4071  | 1.00e-03 | 76 / 5.43 / 3 / 8
    '''
    def __init__(self, log_file_path, hyperparams, aug_counts_list, steps_per_epoch):
        super().__init__()
        self.console = Console()
        self.log_file_path = log_file_path
        self.hyperparams = hyperparams
        self.aug_counts_list = aug_counts_list
        self.steps_per_epoch = steps_per_epoch
        self.table = None

    def on_train_begin(self, logs=None):
        self.table = Table(
            title="[bold blue]DeepAxon++ Training Progress[/bold blue]",
            header_style="bold cyan",
            show_lines=False
        )
        for header in ["Epoch", "Loss", "Acc", "Val_Loss", "Val_Acc", "LR", "Aug (T/A/Mi/Ma)"]:
            self.table.add_column(header, justify="center")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        except Exception:
            lr = float('nan')

        start = epoch * self.steps_per_epoch
        end = min(start + self.steps_per_epoch, len(self.aug_counts_list))
        counts = self.aug_counts_list[start:end]

        if len(counts) > 0:
            total_aug = sum(counts)
            avg_aug = np.mean(counts)
            min_aug = np.min(counts)
            max_aug = np.max(counts)
            aug_str = f"{total_aug:.0f}/{avg_aug:.2f}/{min_aug:.0f}/{max_aug:.0f}"
        else:
            aug_str = "-/-/-/-"

        self.table.add_row(
            str(epoch + 1),
            f"{logs.get('loss', float('nan')):.4f}",
            f"{logs.get('accuracy', float('nan')):.4f}",
            f"{logs.get('val_loss', float('nan')):.4f}",
            f"{logs.get('val_accuracy', float('nan')):.4f}",
            f"{lr:.2e}",
            aug_str
        )

        if (epoch + 1) % 5 == 0 or (epoch + 1) == self.params['epochs']:
            self.console.print(self.table)
            
            
# ===================================================================================== #
#                             MAIN MODEL TRAINER                                        #
# ===================================================================================== #

def train_model(training_dir,
                model_path,
                model_name,
                batch_size=16,
                epochs=200,
                test_fraction=0.3,
                img_ext=".tif",
                mask_ext=".png",
                patch_size=256):
    '''
    Train and save DeepAxon++ model with learning rate scheduler, optional early stopping, and human-readable logs.
    Augmentations are applied dynamically per batch during training.
    
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
    # List to store number of augmented patches per batch
    augmented_counts_per_batch = [] 
    # -------------------- Resolve folders -------------------- #
    images_dir = os.path.join(training_dir, "images")
    masks_dir = os.path.join(training_dir, "masks")

    # -------------------- Load dataset ----------------------- #
    images, masks = load_training_dataset(images_dir, masks_dir, img_ext, mask_ext)

    # -------------------- Crop & patch ----------------------- #
    batch_patch(images, masks, patch_size=patch_size)

    # Patch folders where patches were written
    image_patch_path = os.path.join(images_dir, "cropped", "patches")
    mask_patch_path  = os.path.join(masks_dir, "cropped", "patches")

    # -------------------- Load patches into RAM -------------- #
    X = get_images(image_patch_path)  # returns (n, H, W)
    Y = base_label(get_images(mask_patch_path))

    # Ensure 4D tensors (n_samples, H, W, 1)
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
    if Y.ndim == 3:
        Y = np.expand_dims(Y, axis=-1)
        
    # -------------------- Normalize images -------------------- #
    X = X.astype(np.float32) / 255.0  # normalize training + validation
    
    # -------------------- Dataset summary --------------------- #
    print("\nDataset summary:")

    def list_valid_files(folder, valid_exts=(".png", ".tif", ".tiff", ".jpg", ".jpeg")):
        """
        Return only valid image files from a folder.
        Filters out Thumbs.db, hidden files, and non-image extensions.
        """
        return [
            f for f in os.listdir(folder)
            if (
                os.path.isfile(os.path.join(folder, f))
                and not f.startswith('.')                  # ignore hidden files
                and not f.lower().endswith('thumbs.db')    # ignore Thumbs.db
                and f.lower().endswith(valid_exts)         # only valid image extensions
            )
        ]
        
    def get_image_channels(image_path):
        '''
        Return number of channels for an image (1=grayscale, 3=RGB, 4=RGBA, etc.)
        '''
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "⚠️ unreadable"
        if len(img.shape) == 2:
            return 1
        return img.shape[2]

    # Get valid image and mask files
    image_files = list_valid_files(images_dir)
    mask_files = list_valid_files(masks_dir)

    # Sort both lists alphabetically for alignment
    image_files.sort()
    mask_files.sort()

    # Print original images with channel info
    print(f"  Original images: {len(image_files)}")
    for f in image_files:
        full_path = os.path.join(images_dir, f)
        ch = get_image_channels(full_path)
        print(f"    {f}  ({ch} channel{'s' if isinstance(ch, int) and ch > 1 else ''})")

    # Print original masks with channel info
    print(f"  Original masks:  {len(mask_files)}")
    for f in mask_files:
        full_path = os.path.join(masks_dir, f)
        ch = get_image_channels(full_path)
        print(f"    {f}  ({ch} channel{'s' if isinstance(ch, int) and ch > 1 else ''})")

    # -------- Verify image-mask alignment -------- #
    print("\nVerifying image-mask alignment...")

    # Remove file extensions for comparison
    image_names = [os.path.splitext(f)[0] for f in image_files]
    mask_names = [os.path.splitext(f)[0] for f in mask_files]

    missing_masks = [name for name in image_names if name not in mask_names]
    missing_images = [name for name in mask_names if name not in image_names]

    if not missing_masks and not missing_images:
        print("✅ All images have matching masks.")
    else:
        if missing_masks:
            print("⚠️ Missing masks for:")
            for name in missing_masks:
                print(f"   - {name}")
        if missing_images:
            print("⚠️ Missing images for:")
            for name in missing_images:
                print(f"   - {name}")

    # -------------------- Train / test split ----------------- #
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_fraction, random_state=0)
    
    # -------------------- One-hot encode test masks ---------- #
    y_test_cat = to_categorical(y_test, num_classes=3)

    # -----------------AUG Data Generator --------------------- #
    augmented_counts_per_batch = []
    
    # Soft coded for now, once finalized can integrate into model.py with tensorflow
    def train_generator(X, Y, batch_size, aug_counts_list, **aug_kwargs):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        while True:
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch, Y_batch = X[batch_idx], Y[batch_idx]
                
                # Apply augmentation on-the-fly
                X_aug, Y_aug, flags = augment_dataset_np(X_batch, Y_batch, **aug_kwargs)
                
                # Record augmented count for this batch
                aug_counts_list.append(np.sum(flags))
            
                # One-hot encode masks
                Y_aug_cat = to_categorical(Y_aug, num_classes=3)
                
                yield X_aug, Y_aug_cat

    train_gen = train_generator(X_train, y_train, 
                                batch_size=batch_size,
                                aug_counts_list=augmented_counts_per_batch)
    
    steps_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))   # sum = # patches augmented in this batch
    
    # -------------------- Build model ------------------------ #
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1:4]
    model = deepaxon_plusplus_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ---------------------- Callbacks ------------------------ #
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
    
    # -------------------- Human-readable log ----------------- #
    logs_dir = os.path.join(model_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{model_name}_training_log.txt")

    # Hyperparameters to record at the top of the log
    hyperparams = {
        "Model name": model_name,
        "Start time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Architecture": "UNet++ (DeepAxon++)",
        "Input size (H×W×C)": f"{IMG_HEIGHT}×{IMG_WIDTH}×{IMG_CHANNELS}",
        "Classes": 3,
        "Patch size": patch_size,
        "Batch size": batch_size,
        "Epoch limit": epochs,
        "Test fraction": test_fraction,
        "Augmentations": "H/V flip, small rotation, brightness/contrast, gamma, Gaussian noise (img only)",
        "LR scheduler (ReduceLROnPlateau)": "factor=0.5, patience=30, min_lr=1e-6, monitor=val_loss",
        "EarlyStopping": "patience=60, restore_best_weights=True, monitor=val_loss",
        "Device": "GPU" if use_gpu in ['y','yes'] else "CPU-only",
        "Train samples (orig)": X_train.shape[0],
        "Test samples": X_test.shape[0],
    }

    merged_logger = TrainingLogger(
        log_file_path=log_file,
        hyperparams=hyperparams,
        aug_counts_list=augmented_counts_per_batch,
        steps_per_epoch=steps_per_epoch
    )

    # -------------------- Train ------------------------------ #
    print("[INFO] Starting training with random on-the-fly augmentation...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        verbose=1,
        shuffle=False,
        callbacks=[lr_scheduler, early_stop, merged_logger]
    )

    # -------------------- Save model ------------------------- #
    os.makedirs(model_path, exist_ok=True)
    full_model_path = os.path.join(model_path, model_name + ".keras")
    model.save(full_model_path)

    # -------------------- Print + Log final summary ---------- #
    final_train_acc = history.history.get('accuracy', [None])[-1]
    final_val_acc   = history.history.get('val_accuracy', [None])[-1]
    print(f"\nModel saved at: {full_model_path}")
    print(f"Final training accuracy:    {final_train_acc:.4f}")
    print(f"Final validation accuracy:  {final_val_acc:.4f}")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\nFinal training accuracy:   {final_train_acc:.4f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.4f}\n")

    return model