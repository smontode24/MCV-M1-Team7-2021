import cv2
import numpy as np
from debug_utils import *

def add_kp_args(parser):
    parser.add_argument("--orb_fastthresh", default=15, type=int, help="matching measure in brute force method")
    parser.add_argument("--orb_scaleFactor", default=1.2, type=float)

    # AKAZE Constructor options:
    parser.add_argument("--ak_desc_type", default=5, type=int, help="AKAZE: Descriptor Type. See Ref 0 & ENUM")
    parser.add_argument("--ak_desc_size", default=0, type=int, help="AKAZE: Descriptor size")
    # Valid values: 1, 2 (intensity+gradient magnitude), 3(intensity + X and Y gradients)
    parser.add_argument("--ak_desc_chan", default=3, type=int, help="AKAZE: Descriptor channels")
    parser.add_argument("--ak_threshold", default=0.001, type=float, help=" AKAZE: threshold applied to constructor")
    parser.add_argument("--ak_num_octav", default=4, help="AKAZE: Number of Octaves")
    parser.add_argument("--ak_oct_layer", default=4, help="AKAZE: Number of Octaves layers")
    parser.add_argument("--ak_diffusivt", default=1, help="AKAZE: Diffusivity. See Ref 0 & ENUM")

    parser.add_argument("--sift_features", default=0, type=int, help="SIFT: The number of best features to retain")
    parser.add_argument("--sift_octlayer", default=3, type=int, help="SIFT: Number of Octaves layers")
    parser.add_argument("--sift_thresh", default=0.04, type=float, help=" SIFT: threshold applied to constructor")
    parser.add_argument("--sift_edgethresh", default=10, type=float, help="SIFT: threshold to filter out edge-like "
                                                                          "features")
    parser.add_argument("--sift_sigma", default=1.6, type=float, help="SIFT: reduce the value in case the photos are "
                                                                      "not good quality")

    return parser

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
    
    orb = cv2.ORB_create(fastThreshold=options.orb_fastthresh, scaleFactor=options.orb_scaleFactor) # WTA_K=4, 
    keypoints = orb.detect(grayscale_image, mask=mask)
    return keypoints

def brisk_detect(image, mask, options):
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

    orb = cv2.BRISK_create(thresh=options.brisk_th, patternScale=options.brisk_ps)
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

    # reuse of previous parameters. If not, is complaining because of type
    descriptor_size = int(options.ak_desc_size)
    descriptor_type = int(options.ak_desc_type)
    threshold = float(options.ak_threshold)
    channels = int(options.ak_desc_chan)
    num_octaves = int(options.ak_num_octav)
    octave_layers = int(options.ak_oct_layer)
    difussivity = int(options.ak_diffusivt)
    akaze = cv2.AKAZE_create(descriptor_size=descriptor_size,
                             descriptor_type=descriptor_type,
                             threshold=threshold,
                             descriptor_channels=channels,
                             nOctaves=num_octaves,
                             nOctaveLayers=octave_layers,
                             diffusivity=difussivity)
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

def sift_detect(image, mask, options):
    """
     Extract keypoints and descriptors with Scale Invariant Features Transform
     Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask: mask to be applied to the image [1 = yes, 0 = no]
        options: Optional arguments to adjust the sift option
     Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
        containing local descriptors for the keypoints."""

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask = (mask == 0).astype(np.uint8) * 255

    sift = cv2.SIFT_create(nfeatures=options.sift_features,
                           nOctaveLayers=options.sift_octlayer,
                           contrastThreshold=options.sift_thresh,
                           edgeThreshold=options.sift_edgethresh,
                           sigma=options.sift_sigma)
    keypoints = sift.detect(grayscale_image, mask)

    #drawed_image = cv2.drawKeypoints(z, keypoints, z, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints

# Selection utils
ORB = "ORB"
AKA = "AKAZE"
BRISK = "BRISK"
SIFT = "SIFT"
OPTIONS = [ORB, AKA, BRISK, SIFT]

METHOD_MAPPING = {
    OPTIONS[0]: orb_detect,
    OPTIONS[1]: akaze_detect,
    OPTIONS[2]: brisk_detect,
    OPTIONS[3]: sift_detect,
}

def get_method(method):
    return METHOD_MAPPING[method]