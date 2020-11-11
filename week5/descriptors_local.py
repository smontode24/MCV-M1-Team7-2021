import cv2
import numpy as np

def add_ld_args(parser):
    parser.add_argument("--orb_wtak", default=4, type=int, help="matching measure in brute force method")
    
    parser.add_argument("--brisk_th", default=30, type=int, help="threshold AGAST for brisk")
    parser.add_argument("--brisk_ps", default=1.2, type=float, help="pattern scale of BRISK (scale ratio of radius neighborhood)")
    return parser

def compute_local_desc(image, mask, keypoint, method_name, options):
    method = get_method(method_name)
    return method(image, keypoint, mask, options)

def orb_descriptor(image, kp, mask, options):
    print("descriptors_local.py ==> orb_descriptor")
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

def brisk_descriptor(image, kp, mask, options):
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
    descriptor = orb.compute(grayscale_image, kp)[1]
    return descriptor


def akaze_descriptor(image, kp, mask, options):
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

    # reuse of previous parameters. If not, is complaining
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
    keypoints = akaze.compute(grayscale_image, kp)[1]
    return keypoints

def sift_descriptor(image, kp, mask, options):
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
    descriptors = sift.compute(grayscale_image, kp)[1]

    #drawed_image = cv2.drawKeypoints(z, keypoints, z, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return descriptors


# Selection utils
ORB = "ORB"
AKAZE = "AKAZE"
BRISK = "BRISK"
SIFT = "SIFT"
OPTIONS = [ORB, AKAZE, BRISK, SIFT]

METHOD_MAPPING = {
    OPTIONS[0]: orb_descriptor,
    OPTIONS[1]: akaze_descriptor,
    OPTIONS[2]: brisk_descriptor,
    OPTIONS[3]: sift_descriptor
}

def get_method(method):
    result = METHOD_MAPPING[method]
    return result