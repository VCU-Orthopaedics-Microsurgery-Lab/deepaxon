# ------------------------------ Standard Library --------------------------- #
import os

# ------------------------------ Third-Party Libraries ---------------------- #
# Suppress TensorFlow warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
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
model = load_model(model_path)  #load model from model_path
print("Model loaded successfully!")

# ------------------------------ Process Images ------------------------------ #
# Loop over all files in the input directory and apply segmentation
for img_file in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_file)
    
    # Skip non-image files (optional)
    if not img_file.lower().endswith(('.tif', '.tiff', '.png')):
        print(f"Skipping non-image file: {img_file}")
        continue

    try:
        # Perform segmentation on the image using the specified model
        segment.segment(img_path, model, output_path) #pass loaded model not the path
        print(f"Segmented: {img_file}")
    except Exception as e:
        # Catch errors for individual files without stopping the loop
        print(f"Error processing {img_file}: {e}")