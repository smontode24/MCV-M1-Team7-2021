import numpy as np
import cv2

def estimate_text_mask(imgs, method):
    """ List of list of images. Each list contains one element for each detected painting in the image.
        params:
            imgs: [[painting1, painting2], [painting1], [painting1, painting2], ...]
            method: text segmentation method 
        return: images text boolean mask
    """
    return [[np.zeros(painting).astype(bool) for painting in paintings] for paintings in imgs]

# Define your methods here