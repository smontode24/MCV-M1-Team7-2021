import cv2
import numpy as np
from debug_utils import *

def k_means():
    pass

def BOVW():
    pass



# Selection utils
KMEANS = "KM"
BOW = "BOVW"
OPTIONS = [KMEANS, BOVW,]

METHOD_CLUSTERING = {
    OPTIONS[0]: k_means,
    OPTIONS[1]: BOVW
}

def get_method(method):
    return METHOD_CLUSTERING[method]