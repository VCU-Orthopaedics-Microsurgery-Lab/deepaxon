'''
-------------------------------- DEEPAXON --------------------------------
Main script to train a DeepAxon++ segmentation model.

Folder structure:
training/
├── images/   # names must match masks exactly
└── masks/

The script will crop, patch, load images, train UNet++ model, and save it.
'''

# ------------------------------ Standard Library ------------------------------ #
import os                             

# ------------------------------ Local Imports --------------------------------- #
import train                          # DeepAxon training module

# ------------------------------ Helper Functions ------------------------------ #
def get_valid_path(prompt):
    '''
    Ask user for a folder path until a valid one is provided.
    '''
    while True:
        path = input(prompt).strip()
        if os.path.isdir(path):
            return path
        else:
            print(f"Path not found: {path}. Please enter a valid folder path.")

def get_training_dir():
    '''
    Ask for the training folder and ensure it has 'images' and 'masks' subfolders.
    '''
    while True:
        training_dir = get_valid_path("Input the path to the training folder that holds the images and masks: ")
        images_dir = os.path.join(training_dir, "images")
        masks_dir = os.path.join(training_dir, "masks")
        if not os.path.isdir(images_dir):
            print(f"'images' subfolder not found in {training_dir}. Please check the folder structure.")
            continue
        if not os.path.isdir(masks_dir):
            print(f"'masks' subfolder not found in {training_dir}. Please check the folder structure.")
            continue
        return training_dir

def get_model_dir():
    '''
    Ask for a model save folder and create it if it doesn't exist.
    '''
    while True:
        model_dir = input("Input the path of the folder where the model will be saved: ").strip()
        if not model_dir:
            model_dir = os.getcwd()
            print(f"No path entered. Using current directory: {model_dir}")
        try:
            os.makedirs(model_dir, exist_ok=True)
            return model_dir
        except Exception as e:
            print(f"Failed to create/access folder: {e}")

def get_int_input(prompt, default):
    '''
    Ask for a positive integer input, with default if empty.
    '''
    while True:
        user_input = input(f"{prompt} (press Enter for default={default}): ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if value <= 0:
                print("Please enter a positive integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")
            
def get_float_input(prompt, default):
    '''
    Ask the user for a float input. Return default if empty.
    Ensures the value is between 0 and 1.
    '''
    while True:
        user_input = input(f"{prompt} (press Enter for default={default}): ").strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if not 0 < value < 1:
                print("Please enter a decimal number between 0 and 1 (e.g., 0.3 for 30%).")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a decimal number.")            

# ------------------------------ Main Prompts -------------------------------- #
training_dir = get_training_dir()
model_dir = get_model_dir()

# Suggest an example model name
model_name = input("Input the name of the model (e.g., Rb_100x_proximal): ").strip()
if not model_name:
    print("No model name entered. Exiting.")
    exit()

# Numeric inputs with defaults
batch_size = get_int_input("Batch size", 16)
epochs = get_int_input("Epochs", 200)
test_size = get_float_input("Fraction of dataset to reserve for testing (0-1)", 0.3)

# ------------------------------ Start Training ------------------------------ #
print(f"\nTraining model '{model_name}' with batch size={batch_size}, epochs={epochs}, train/test split={test_size}...")
train.train_model(training_dir, model_dir, model_name, batch_size, epochs,test_fraction=test_size)
