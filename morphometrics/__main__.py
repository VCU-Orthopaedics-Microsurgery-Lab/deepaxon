# ------------------------------ Standard Library ---------------------------- #
import os

# ------------------------------ Local Imports ------------------------------- #
import morphometrics

# ------------------------------ Input Paths --------------------------------- #
dir_path = input("Input the path to the folder of segmented images: ")
output_dir = input("Input the path to the output folder: ")

# ------------------------------ Main Loop ----------------------------------- #
for img_name in os.listdir(dir_path):
    # Skip non-image files
    if not img_name.lower().endswith(('.tif', '.tiff', '.png')):
        continue
    if img_name == 'Thumbs.db':
        print("Ignoring 'Thumbs.db'")
        continue
    # Process image
    print(f'Obtaining morphometrics for {img_name}')
    img_path = os.path.join(dir_path, img_name)
    output_name = img_name.split('.')[0]
    morph_df = morphometrics.get_morphometrics(img_path)
    morphometrics.save_morphometrics(morph_df, output_dir, output_name)