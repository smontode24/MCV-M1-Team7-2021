import cv2
import numpy as np
from debug_utils import *

def k_means():
    pass

def shift_mean():
    pass

def bovw():
    pass



# Selection utils
KMEANS = "KM"
BOVW = "BOVW"
SHIFT_MEAN = "SHIFT_MEAN"
OPTIONS = [KMEANS, SHIFT_MEAN, BOVW]

METHOD_CLUSTERING = {
    OPTIONS[0]: k_means,
    OPTIONS[1]: shift_mean,
    OPTIONS[2]: bovw,
}

def get_method(method):
    return METHOD_CLUSTERING[method]