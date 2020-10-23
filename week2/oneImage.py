import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from text_segmentation import *
from metrics import *
from evaluation.mask_evaluation import *
from evaluation.retrieval_evaluation import *
from time import time
from io_utils import *
from debug_utils import *
from scipy import ndimage
import numpy as np


# RETURN: Given a relative path, it return it's absolute
def absolutePath(relative):
    # Join of a first system separator + PWD function + Relative = Abosulte path
    return os.path.join(os.path.sep, os.path.dirname(__file__), relative)

# RETURN: An image specified on the path (absolute or relative)
def openImage(path):
    img = cv2.imread(path)

    return img

# IMG = 3 channel RGB image (ndarray)
# RETURN: An image with 1 dimension, gray
def rgb2gray(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_image


def rgb2gray2(img):
    gray_image = img.dot([0.07, 0.72, 0.21])

    return gray_image

# RETURN: An image with its gradient in X dimension
def gradientX (img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)

def gradientY (img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)

# Veure gradients x i y, y obtenir els dos rectangles més grans


# Reusing some calls from the MAIN code,
# this program focused in one imag
if __name__ == "__main__":
    # Path to DB image:
    path0 = "BBDD/bbdd_00219.jpg"
    path0 = absolutePath(path0)
    print ("Image from the DB: ", path0)
    # Path to image in QDS1
    path1 = "qsd1_w2/00004.jpg"
    path1 = absolutePath(path1)
    # Path to image in QDS2
    path2 = "qsd2_w2/00000.jpg"
    path2 = absolutePath(path2)
    # Path to mask in QSD2
    path3 = "qsd2_w2/00000.png"
    path3 = absolutePath(path3)

    img0 = openImage(path0)
    img1 = openImage(path1)
    img2 = openImage(path2)
    msk2 = openImage(path3)

    #Extracted from here: https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python
    gradX_0 = ndimage.sobel(img0, axis=0, mode='constant')
    gradY_0 = ndimage.sobel(img0, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_0 = np.hypot(gradX_0, gradY_0)

    gradX_1 = ndimage.sobel(img1, axis=0, mode='constant')
    gradY_1 = ndimage.sobel(img1, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_1 = np.hypot(gradX_1, gradY_1)

    gradX_2 = ndimage.sobel(img2, axis=0, mode='constant')
    gradY_2 = ndimage.sobel(img2, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_2 = np.hypot(gradX_2, gradY_2)

    gradX_m2 = ndimage.sobel(msk2, axis=0, mode='constant')
    gradY_m2 = ndimage.sobel(msk2, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_m2 = np.hypot(gradX_m2, gradY_m2)

    ## IMAGES IN BW
    img0_bw = rgb2gray(img0)
    img1_bw = rgb2gray(img1)
    img2_bw = rgb2gray(img2)
    msk2_bw = rgb2gray(msk2)

    ## GRADIENTS IN BW  [USING NUMPY]
    gradX_0_bw = ndimage.sobel(img0_bw, axis=0, mode='constant')
    gradY_0_bw = ndimage.sobel(img0_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_0_bw = np.hypot(gradX_0_bw, gradY_0_bw)

    gradX_1_bw = ndimage.sobel(img1_bw, axis=0, mode='constant')
    gradY_1_bw = ndimage.sobel(img1_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_1_bw = np.hypot(gradX_1_bw, gradY_1_bw)

    gradX_2_bw = ndimage.sobel(img2_bw, axis=0, mode='constant')
    gradY_2_bw = ndimage.sobel(img2_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_2_bw = np.hypot(gradX_2_bw, gradY_2_bw)

    gradX_m2_bw = ndimage.sobel(msk2_bw, axis=0, mode='constant')
    gradY_m2_bw = ndimage.sobel(msk2_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_m2_bw = np.hypot(gradX_m2_bw, gradY_m2_bw)


    ## GRADIENTS IN BW  [USING CV]
    cv_gradX_0_bw = gradientX(img0_bw)
    cv_gradY_0_bw = gradientY(img0_bw)
    # Get square root of sum of squares
    cv_sobel_0_bw = np.hypot(cv_gradX_0_bw, cv_gradY_0_bw)

    cv_gradX_1_bw = gradientX(img1_bw)
    cv_gradY_1_bw = gradientY(img1_bw)
    # Get square root of sum of squares
    cv_sobel_1_bw = np.hypot(cv_gradX_1_bw, cv_gradY_1_bw)

    cv_gradX_2_bw = gradientX(img2_bw)
    cv_gradY_2_bw = gradientY(img2_bw)
    # Get square root of sum of squares
    cv_sobel_2_bw = np.hypot(cv_gradX_2_bw, cv_gradY_2_bw)

    cv_gradX_m2_bw = ndimage.sobel(msk2, axis=0, mode='constant')
    cv_gradY_m2_bw = ndimage.sobel(msk2, axis=1, mode='constant')
    # Get square root of sum of squares
    cv_sobel_m2_bw = np.hypot(cv_gradX_m2_bw, cv_gradY_m2_bw)

    awesomeImage = cv_sobel_2_bw



    print("hey, stop!")
