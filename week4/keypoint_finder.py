import cv2
import numpy as np
from debug_utils import *

def compute_keypoints(image, mask, method_name, options):
    method = get_method(method_name)
    kp = method(image, mask, options)

    if isDebug():
        img_copy = cv2.resize(image.copy(), (256,256))
        img_copy = cv2.drawKeypoints(img_copy, kp, None, color=(0,255,0), flags=0)
        cv2.imshow("detected keypoints", img_copy)
        cv2.waitKey(0)
    return kp

def orb_detect(image, mask, options):
    """
    Extract descriptors from image using the ORB method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): list of cv2.KeyPoint objects.
    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
            containing local descriptors for the keypoints.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask = (mask==0).astype(np.uint8)*255

    orb = cv2.ORB_create(WTA_K=4, fastThreshold=15)
    keypoints = orb.detect(grayscale_image, mask=mask)
    return keypoints

# Selection utils
ORB = "ORB"
OPTIONS = [ORB]

METHOD_MAPPING = {
    OPTIONS[0]: orb_detect
}

def get_method(method):
    return METHOD_MAPPING[method]