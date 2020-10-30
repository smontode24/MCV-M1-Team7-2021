import cv2
import numpy as np
from debug_utils import *

def denoise_images(qs_imgs, method_name="median"):
    """ Denoise a list of images """
    if method_name == "none":
        return qs_imgs
    
    noise_removal_method = get_noise_removal_method(method_name) 
    denoised_imgs = [denoise_median(qs_img) for qs_img in qs_imgs]

    if isDebug():
        for denoised_img, qs_img in zip(denoised_imgs, qs_imgs):
            cv2.imshow("original image", qs_img)
            cv2.imshow("denoised image", denoised_img)
            cv2.imshow("difference", np.abs(qs_img-denoised_img).astype(np.uint8))
            cv2.waitKey(0)
    
    return denoised_imgs

def denoise_median(img):
    return cv2.medianBlur(img, 5)

def denoise_bilateral(img, d=7, sigmaColor=75, sigmaSpace=75):
    """
    Bilateral filter: Filter that preserves edges well, the rest is smoothed with a gaussian.
    d: Diameter of each pixel neighborhood.
    sigmaColor: Value of \sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
    sigmaSpace: Value of \sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their
    colors lie within the sigmaColor range.
    """
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def denoise_gaussian(img, k=3, sigma=2):
    """
    Bilateral filter: Filter that preserves edges well, the rest is smoothed with a gaussian.
    sigma: Diameter of each pixel neighborhood.
    """
    return cv2.GaussianBlur(img, k, sigma) 

noise_removal_methods = {
    "median": denoise_median,
    "bilateral": denoise_bilateral,
    "gaussian": denoise_gaussian
}


def get_noise_removal_method(method_name):
    return noise_removal_methods[method_name]