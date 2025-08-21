import os
import cv2

def resize_img(img_path):
    img = cv2.imread(img_path, 0)
    if img.shape == (2880, 2048):
        return cv2.resize(img, (1440, 1024))
    return img