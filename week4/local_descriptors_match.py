import skimage
from skimage.feature import match_descriptors
import numpy as np
import cv2

def argparser_add_opts(parser):
    parser.add_argument("--bf_metric", default="hamming", type=str, help="matching measure in brute force method")
    parser.add_argument("--bf_max_ratio", default=0.8, type=float, help="matching measure in brute force method")
    return parser

def match_descriptors_qs_db(qs_descriptors, db_descriptors, method_name, options):
    method = get_method(method_name)
    result = [[method(qs_descriptor, db_descriptor, options) for db_descriptor in db_descriptors] for qs_descriptor in qs_descriptors]
    result = np.array(result)
    return result

def automatic_brute_force_match(descriptors1, descriptors2, options):
    """ Returns number of matches for descriptor"""
    if type(descriptors1) != np.ndarray or type(descriptors2) != np.ndarray:
        return 0
    
    metric = options.bf_metric 
    max_ratio = options.bf_max_ratio
    matches = match_descriptors(descriptors1, descriptors2, metric=metric, max_ratio=max_ratio)
    # TODO: Add visualization tool if debug is enabled for point correspondance
    return len(matches)

def auto_bf_matcher_cv(descriptors1, descriptors2, options):
    """ Returns number of matches for descriptor"""
    if type(descriptors1) != np.ndarray or type(descriptors2) != np.ndarray:
        return 0

    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING2, crossCheck=False) 
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
    return len(good)

""" 
When using brote force matcher. From OpenCV documentation:
*---
One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are preferable choices for SIFT and SURF descriptors, 
NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description).
*---
"""

# Selection utils
BRUTE_FORCE = "BF"
OPTIONS = [BRUTE_FORCE]

METHOD_MAPPING = {
    OPTIONS[0]: automatic_brute_force_match
}

def get_method(method):
    return METHOD_MAPPING[method]
