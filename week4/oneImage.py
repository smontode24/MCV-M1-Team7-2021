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
from matplotlib import pyplot as plt
import math


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

# RETURN: An image with its gradient in X dimension
def gradientX (img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)

# RETURN: An image with its gradient in Y dimension
def gradientY (img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Credits: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections



# Reusing some calls from the MAIN code,
# this program focused in one imag
if __name__ == "__main__":
    # Path to DB image:
    path0 = "BBDD/bbdd_00164.jpg"
    path0 = absolutePath(path0)
    print ("Image from the DB: ", path0)
    # Path to image in DB
    path1 = "qsd1_w4/00017.jpg"
    path1 = absolutePath(path1)
    # Path to image not in DB
    path2 = "qsd1_w4/00020.jpg"
    path2 = absolutePath(path2)


    imgDB = openImage(path0)
    imgIN = openImage(path1)
    imgNOT = openImage(path2)


    cv2.imshow("Image from the DB", imgDB)
    cv2.imshow("Image IN the DB", imgIN)
    cv2.imshow("Image NOT in the DB", imgNOT)

    cv2.waitKey()

    print("hey, stop!")
