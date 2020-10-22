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
import numpy as np

# RETURN: Given a relative path, it return it's absolute
def absolutePath(relative):
    # Join of a first system separator + PWD function + Relative = Abosulte path
    return os.path.join(os.path.sep, os.path.dirname(__file__), relative)

# RETURN: An image specified on the path (absolute or relative)
def openImage(path):
    img = cv2.imread(path)

    return img

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

    print("hey, stop!")
