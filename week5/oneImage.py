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
from operator import itemgetter

# Own coded comparator, in order to order nested lists in Python
def compareDistances (dMatch):
    return dMatch.distance

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
    path3 = "qsd1_w4/00017.png"
    path3 = absolutePath(path3)


    imgDB = openImage(path0)
    imgIN = openImage(path1)
    imgNOT = openImage(path2)


    #cv2.imshow("Image from the DB", imgDB)
    #cv2.imshow("Image IN the DB", imgIN)
    #cv2.imshow("Image NOT in the DB", imgNOT)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Create self-made mask
    # Setting up directly the points of the text
    mask1 = openImage(path3)
    # Modifying mask1 and deleting textbox
    cv2.rectangle(mask1, (216,732),(606,825),(0,0,0),-1)

    # Find keypoints for each image
    kp1 = orb.detect(imgDB,None)
    kp2 = orb.detect(imgIN, mask=mask1) # << This can be [inputs and masks ... arrays]
    kp3 = orb.detect(imgNOT,None)

    # Compute de descriptor
    kp1, des1 = orb.compute(imgDB, kp1)
    kp2, des2 = orb.compute(imgIN, kp2)
    kp3, des3 = orb.compute(imgNOT, kp3)

    imgDB_drawn = cv2.drawKeypoints(imgDB, kp1, None, color=(0,255,0), flags=0)
    imgIN_drawn = cv2.drawKeypoints(imgIN, kp2, None, color=(0,255,0), flags=0)
    imgNOT_drawn = cv2.drawKeypoints(imgNOT, kp3, None, color=(0,255,0), flags=0)

    #cv2.imshow("Image from the DB", imgDB_drawn)
    #cv2.imshow("Image IN the DB", imgIN_drawn)
    #cv2.imshow("Image NOT in the DB", imgNOT_drawn)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches1 = bf.match(des1, des2)

    ordered_matches = sorted(matches1, key=compareDistances)
    #TODO: Create a vector ordered by distance

    # PART 2:
    # Tell the descriptor not to use points in mask
    img3 = cv2.drawMatches(imgDB, kp1, imgIN, kp2, matches1[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img4 = cv2.drawMatches(imgDB, kp1, imgIN, kp2, ordered_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matching", img3)
    cv2.imshow("Matching ordered", img4)
    cv2.waitKey()

    print("hey, stop!")
