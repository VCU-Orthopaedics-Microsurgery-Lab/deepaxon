# ------------------------------ Standard Library --------------------------- #
import os
import datetime

# ------------------------------ Third-Party Libraries ---------------------- #
# Suppress TensorFlow warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.models import load_model

# ------------------------------ Local Imports ------------------------------- #
import segment

# ------------------------------ GPU / CUDA Check ---------------------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected ✅")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")
else:
    print("No GPU detected, running on CPU.")

# ------------------------------ Custom Metrics and Loss ------------------------- #
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
    
# ------------------------------ User Input Paths ---------------------------- #
# Prompt user for paths to the model, input folder, and output folder
model_path = input("Input the path to the model: ")
dir_path = input("Input the path to the folder of images: ")
output_path = input("Input the path for the output folder: ")

# ------------------------------ Validate Paths ------------------------------ #
# Ensure the model file and input folder exist; create output folder if needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(dir_path):
    raise FileNotFoundError(f"Input folder not found: {dir_path}")
os.makedirs(output_path, exist_ok=True)

# ------------------------------ Load Model Once ----------------------------- #
print("Loading model...")
model = load_model(model_path, custom_objects={
    "combined_loss": combined_loss,
    "dice_coef": dice_coef,
    "iou_coef": iou_coef
})
print("Model loaded successfully!")

# ------------------------------ Process Folder ------------------------------ #
print("Segmenting folder...")
times = segment.segment_dir(dir_path, model, output_path)