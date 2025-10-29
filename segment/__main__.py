# ------------------------------ Standard Library --------------------------- #
import os

# ------------------------------ Local Imports ------------------------------- #
import segment

# ------------------------------ User Input Paths ---------------------------- #
# Prompt user for paths to the model, input images folder, and output folder
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
        segment.segment(img_path, model_path, output_path)
        print(f"Processed: {img_file}")
    except Exception as e:
        # Catch errors for individual files without stopping the loop
        print(f"Error processing {img_file}: {e}")