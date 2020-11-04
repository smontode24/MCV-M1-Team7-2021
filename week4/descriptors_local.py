import cv2
import numpy as np

def add_ld_args(parser):
    parser.add_argument("--orb_wtak", default=4, type=int, help="matching measure in brute force method")
    return parser

def compute_local_desc(image, mask, keypoint, method_name, options):
    method = get_method(method_name)
    return method(image, keypoint, mask, options)

def orb_descriptor(image, kp, mask, options):
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

    orb = cv2.ORB_create(WTA_K=options.orb_wtak)
    descriptor = orb.compute(grayscale_image, kp)[1]
    return descriptor

# Selection utils
ORB = "ORB"
OPTIONS = [ORB]

METHOD_MAPPING = {
    OPTIONS[0]: orb_descriptor
}

def get_method(method):
    return METHOD_MAPPING[method]