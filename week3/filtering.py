import cv2

def denoise_images(qs_imgs, method_name="median"):
    """ Denoise a list of images """
    noise_removal_method = get_noise_removal_method(method_name) 
    return [denoise_median(qs_img) for qs_img in qs_imgs]

def denoise_median(img):
    return cv2.medianBlur(img, 5)

noise_removal_methods = {
    "median": denoise_median
}


def get_noise_removal_method(method_name):
    return noise_removal_methods[method_name]