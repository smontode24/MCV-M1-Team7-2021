import cv2
import numpy as np
from debug_utils import *

def add_kp_args(parser):
    parser.add_argument("--orb_fastthresh", default=15, type=int, help="matching measure in brute force method")

    # AKAZE Constructor options:
    parser.add_argument("--ak_desc_type", default=5, type=int, help="AKAZE: Descriptor Type. See Ref 0 & ENUM")
    parser.add_argument("--ak_desc_size", default=3, type=int, help="AKAZE: Descriptor size")
    parser.add_argument("--ak_desc_chan", default=3, type=int, help="AKAZE: Descriptor channels")
    parser.add_argument("--ak_threshold", default=0.001, type=float, help=" AKAZE: threshold applied to constructor")
    parser.add_argument("--ak_num_octav", default=4, help="AKAZE: Number of Octaves")
    parser.add_argument("--ak_oct_layer", default=4, help="AKAZE: Number of Octaves layers")
    parser.add_argument("--ak_diffusivt", default=3, help="AKAZE: Diffusivity. See Ref 0 & ENUM")


    return parser

def compute_keypoints(image, mask, method_name, options):
    print("keypoint_finder.py ==> compute_keypoints")
    method = get_method(method_name)
    kp = method(image, mask, options)

    if isDebug():
        img_copy = cv2.resize(image.copy(), (256,256))
        img_copy = cv2.drawKeypoints(img_copy, kp, None, color=(0,255,0), flags=0)
        cv2.imshow("detected keypoints", img_copy)
        cv2.waitKey(0)
    return kp

def orb_detect(image, mask, options):
    print("keypoint_finder.py ==> orb_detect")
    """
    Extract descriptors from image using the ORB method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask: mask to be applied to the image [1 = yes, 0 = no]
        options: Optional arguments to adjust the orb_detect option
    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
            containing local descriptors for the keypoints.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask = (mask==0).astype(np.uint8)*255

    orb = cv2.ORB_create(fastThreshold=options.orb_fastthresh) # WTA_K=4, 
    keypoints = orb.detect(grayscale_image, mask=mask)
    return keypoints


def akaze_detect(image, mask, options):
    """
    Extract descriptors from image using the AKAZE method.
    # REF 0: https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html
    --
    # REF 1: https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
    # REF 2: http://man.hubwiz.com/docset/OpenCV.docset/Contents/Resources/Documents/db/d70/tutorial_akaze_matching.html
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask: mask to be applied to the image [1 = yes, 0 = no]
        options: Optional arguments to adjust the akaze option
    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
            containing local descriptors for the keypoints.
    """
    print("keypoint_finder.py ==> akaze_detect")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask = (mask == 0).astype(np.uint8) * 255

    # Options of the constructor
    #    - descriptor_type			Type of the extracted descriptor: DESCRIPTOR_KAZE,
    #                               DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    #    - descriptor_size          Size of the descriptor in bits. 0 -> Full size
    #    - descriptor_channels      Number of channels in the descriptor (1, 2, 3)
    #    - threshold                Detector response threshold to accept point
    #    - nOctaves                 Maximum octave evolution of the image
    #    - nOctaveLayers            Default number of sublevels per scale level
    #    - diffusivity              Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

    akaze = cv2.AKAZE_create(descriptor_size=options.ak_desc_size,
                             descriptor_type=options.ak_desc_type,
                             threshold=options.ak_threshold,
                             descriptor_channels=options.ak_desc_chan,
                             nOctaves=options.ak_num_octav,
                             nOctaveLayers=options.ak_oct_layer,
                             diffusivity=options.ak_diffusivt)
    keypoints = akaze.detect(grayscale_image, mask=mask)
    return keypoints



'''
FOR THE FUTURE:::: AS IT SEEMS IS WORSE THAN AKAZE
def kaze_detect(image, mask, options):
    """
    Extract descriptors from image using the KAZE method.
    # REF 0: https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html
    --
    # REF 1: https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
    # REF 2: http://man.hubwiz.com/docset/OpenCV.docset/Contents/Resources/Documents/db/d70/tutorial_akaze_matching.html
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask: mask to be applied to the image [1 = yes, 0 = no]
        options: Optional arguments to adjust the akaze option
    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
            containing local descriptors for the keypoints.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask = (mask==0).astype(np.uint8)*255

    ## Options of the constructor
    #    - extended:      Set to enable extraction of extended (128-byte) descriptor.
    #    - upright: 	  Set to enable use of upright descriptors (non rotation-invariant).
    #    - threshold:     Detector response threshold to accept point
    #    - nOctaves:      Maximum octave evolution of the image
    #    - nOctaveLayers: Default number of sublevels per scale level
    #    - diffusivity:   Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
    kaze = cv2.KAZE_create(threshold=options.akaze_threshold)
    keypoints = kaze.detect(grayscale_image, mask=mask)
    return keypoints
'''

# Selection utils
ORB = "ORB"
AKA = "akaze"
OPTIONS = [ORB, AKA]

METHOD_MAPPING = {
    OPTIONS[0]: orb_detect,
    OPTIONS[1]: akaze_detect
}

def get_method(method):
    return METHOD_MAPPING[method]