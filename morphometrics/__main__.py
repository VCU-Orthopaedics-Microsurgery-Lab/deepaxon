import os
import morphometrics

dir_path = input("Input the path to the folder of segmented images: ")
output_dir = input("Input the path to the output folder: ")

for img_name in os.listdir(dir_path):
    if img_name == 'Thumbs.db':
        print("Ignoring 'Thumbs.db'")
    else:
        print(f'Obtaining morphometrics for {img_name}')
        img_path = os.path.join(dir_path, img_name)
        output_name = img_name.split('.')[0]
        morph_df = morphometrics.get_morphometrics(img_path)
        morphometrics.save_morphometrics(morph_df, output_dir, output_name)