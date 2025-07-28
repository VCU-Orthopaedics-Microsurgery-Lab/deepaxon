'''
-------------------------------- DEEPAXON --------------------------------
Train a segmentation model using the DeepAxon++ model. Training folder should follow the folder structure of:

training
|__images
   |__image1.png
   |__image2.png
   |__ . . .
|__masks
   |__image1.png
   |__image2.png
   |___ . . .
   
can specifiy the name and the folder to save it in
'''

from keras.utils import normalize, to_categorical #keras dependencies for normalizing and categorizing datasets
import os #to work with files and directories
import cv2 #to work with images
import numpy as np
from sklearn.preprocessing import LabelEncoder #to relabel masks into 0,1,2
from sklearn.model_selection import train_test_split #create 70/30 train test split
from model import deepaxon_plusplus_model #model
from PIL import Image #for image-related work
from patchify import patchify #for simple patch making

def cropper(image_path):
    '''
    Crops an image into a square, cutting equally from any side that has to be cut
    
    :param image_path: A path (string or object) that directs to a single image to be cropped
    
    :returns: A PIL Image object that is the centered, cropped image
    '''
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #opens the given image

    SIZE_X = image.shape[1] // 256 * 256
    SIZE_Y = image.shape[0] // 256 * 256
    
    image = Image.fromarray(image)
    cropped = image.crop((0,0,SIZE_X, SIZE_Y))

    return cropped

def dir_crop(dir_path):
    '''
    Crops every image in a directory and saves then in a subfolder called 'cropped' in the given directory
    
    :param dir_path: A path (string or object) that directs to a directory with one or more images. The directory should only contain images
    
    :returns: The path of the subfolder named 'cropped' which was made in the given folder
    '''
    
    #make a subfolder called 'cropped'
    crop_dir = os.path.join(dir_path,'cropped')
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
    
    #go through every image in the given folder and save a cropped version in the new 'cropped' subfolder
    for image_name in os.listdir(dir_path):
        
        filetype = len(image_name.split('.'))
        
        #only run for files, not folders
        if filetype == 2:
            image_path = os.path.join(dir_path, image_name)
            crop_path = os.path.join(dir_path,'cropped',image_name)
            cropped = cropper(image_path)
            cropped.save(crop_path)
    
    return crop_dir

def patch(image_path):
    '''
    Take input of a single image path. Save 256x256 patches to a subfolder called 'patches' for the single image.
    The patches will be B&W png images regardless of the initial picture format.
    
    :param image_path: A path (string or object) that points to a single image to be turned into patches
    
    :returns: Nothing
    '''
    
    #extract information about the image name and path
    image_name = os.path.basename(image_path).split('.')[0] #get the name of the image. eg. 'train/images/cropped/image1.png' --> 'image1'
    image_root = os.path.dirname(image_path) #get the path to the directory the image is in. eg. 'train/images/cropped/image1.png' --> 'train/image/cropped/'
    
    #make a subfolder named 'patches' in the root folder of the image
    patch_dir = os.path.join(image_root,'patches')
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)

    image = cv2.imread(image_path)
    print(image_path)
    if len(image.shape) == 3 and image.shape[2] == 3:
       image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts the image to black-and-white but does not save the image anywhere. just used for processing
    
    patches_img = patchify(image, (256,256), step=256) #using patchify to make patches
    
    #saves all patches in the new patches folder
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):    
            single_patch_img = patches_img[i,j,:,:]
            patch_path = os.path.join(patch_dir, image_name+'_'+str(i)+str(j)+'.png')
            cv2.imwrite(patch_path, single_patch_img)

def dir_patch(dir_path):
   '''
   Takes input of a folder of images. In the proper setup for patch, this program will be run for either the images or masks folder.
   Turns every photo in the folder into patches and saves the patches in a new 'patches' folder.
   Uses DeepAxon.patch function patch() to generation patches.

   :param dir_path: A path (string or object) that points to a folder of images to convert every image in the folder into patches

   :returns: Nothing
   '''
    
   #goes through every image in the folder and runs the DeepAxon.patch function patch()
   for image_name in os.listdir(dir_path):
      image_path = os.path.join(dir_path, image_name)
      patch(image_path)   
        
def batch_patch(dir_path):
    '''
    Takes input of a training folder that follows the following setup:\n
    training\n\timages\n\t\timage1.png\n\n\t\timage2.png\n\tmasks\n\t\timage1.png\n\n\t\timage2.png
    
    The folder called 'training' should be the parameter. For both the images and masks subfolders, this function will create a folder of cropped images.
    Inside the folder of copped images, a subfolder of 256x256 patches will be made. The new folder structure will be as follows:\n
    training\n\timages\n\t\tcropped\n\t\t\tpatches\n\t\t\t\timage1_01.png\n\n\t\t\t\timage1_02.png . . .\n\t\t\timage1.png\n\n\t\t\timage2.png
    \n\t\timage1.png\n\n\t\timage2.png\n\tmasks . . .
    
    :param dir_path: A path (string or object) that points to a training folder that follows the above structuring
    
    :returns: Nothing
    '''
    
    #make variables for the path the the images and masks folders
    image_path = os.path.join(dir_path, 'images')
    mask_path = os.path.join(dir_path, 'masks')
    
    #make the cropped folders with cropped images and masks
    image_path = dir_crop(image_path)
    mask_path = dir_crop(mask_path)
    
    #make patches for the cropped images and masks
    dir_patch(image_path)
    dir_patch(mask_path)

def get_images(patch_path):
    '''
    Get all the images in a folder as a numpy array
    
    :param patch_path: A path (string or object) 
    
    :returns: Numpy Array; all images in the patches folder
    '''
    
    #go to every list in the patch folder and append it to the list
    img_array = []
    for image_name in os.listdir(patch_path):
        image_path = os.path.join(patch_path, image_name)
        img = cv2.imread(image_path, 0)
        img_array.append(img)

    #convert 
    return np.array(img_array)

def base_label(train_masks):
    '''
    If the manual masks are made with colors instead of 0,1,2 this function will convert it into 0,1,2.
    
    :param train_masks: Numpy Array; training mask images
    
    :returns: Numpy Array; training mask images with values being 0,1,2 instead of whatever colors it was
    '''
    
    pixel_to_class = {0: 0, 127: 1, 128: 1, 255: 2}
    train_masks_remapped = np.vectorize(lambda x: pixel_to_class.get(x, 0))(train_masks)
    return train_masks_remapped


def train_model(dir_path, model_path, model_name, batch_size=16, epochs=200):
    '''
    Train and save model from training images and masks. Uses DeepAxon++ architecture
    
    :param dir_path: A path (string or object) to a training folder which had both images and masks folders
    :param model_path: The path to the directory to save the model in. Recommend making a model folder to save models in
    :param model_name: Name of the model to be save as
    
    :returns: Keras model
    '''
    
    batch_patch(dir_path) #create patches folder for training images and masks
    
    #get the paths to the image and mask patches
    image_patch_path = os.path.join(dir_path, 'images', 'cropped', 'patches')
    mask_patch_path = os.path.join(dir_path, 'masks', 'cropped', 'patches')
    
    #get the images and masks as numpy arrays
    train_images = normalize(get_images(image_patch_path), axis=1)
    train_masks = base_label(get_images(mask_patch_path))
    
    #add another dimension to the arrays for training purposes
    train_images = np.expand_dims(train_images, axis=3)
    train_masks = np.expand_dims(train_masks, axis=3)
    
    #70/30 training/testing split
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size = 0.10, random_state = 0)
    
    #convert training masks into categorical variables
    train_masks_cat = to_categorical(y_train, num_classes=3)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], 3)) #last dimension is 3 for background, axon, myelin
    
    #convert test masks into categorical variables
    test_masks_cat = to_categorical(y_test, num_classes=3)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], 3)) #last dimension is 3 for background, axon, myelin
    
    #get shape of training set
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    
    model = deepaxon_plusplus_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=3) #obtain model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compile architecture
    model.fit(X_train, y_train_cat, batch_size=batch_size, verbose=1, epochs=epochs, validation_data=(X_test, y_test_cat), shuffle=False) #train model
    
    model.save(os.path.join(model_path, model_name+'.keras')) #save model