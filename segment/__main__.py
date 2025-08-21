import segment

model_path = input("Input the path to the model: ")
dir_path = input("Input the path to the folder of images: ")
output_path = input("Input the path for the output folder: ")

segment.segment_dir(dir_path, model_path, output_path)