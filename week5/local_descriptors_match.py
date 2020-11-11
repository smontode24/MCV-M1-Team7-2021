import skimage
from skimage.feature import match_descriptors
import numpy as np
import cv2
from debug_utils import *
import matplotlib.pyplot as plt
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from tqdm import tqdm

def add_md_args(parser):
    parser.add_argument("--bf_metric", default="hamming", type=str, help="matching measure in brute force method")
    parser.add_argument("--bf_max_ratio", default=0.8, type=float, help="matching measure in brute force method")
    return parser

def match_descriptors_qs_db(qs_imgs, db_imgs, qs_descriptors, db_descriptors, qs_kps, db_kps, method_name, options):
    method = get_method(method_name)
    result = [[method(db_img, qs_img, qs_descriptor, db_descriptor, qs_kp, db_kp, options) for db_descriptor, db_img, db_kp in zip(db_descriptors, db_imgs, db_kps)] for qs_descriptor, qs_img, qs_kp in zip(qs_descriptors, qs_imgs, qs_kps)]
    #for qs_descriptor, qs_img, qs_kp in tqdm(zip(qs_descriptors, qs_imgs, qs_kps)):
    #result.append([[method(db_img, qs_img, qs_descriptor, db_descriptor, qs_kp, db_kp, options) for db_descriptor, db_img, db_kp in zip(db_descriptors, db_imgs, db_kps)] for qs_descriptor, qs_img, qs_kp in zip(qs_descriptors, qs_imgs, qs_kps)]) #[method(db_img, qs_img, qs_descriptor, db_descriptor, qs_kp, db_kp, options) for db_descriptor, db_img, db_kp in zip(db_descriptors, db_imgs, db_kps)])
    #result = [ ]
    result = np.array(result)
    return result

def automatic_brute_force_match(db_img, qs_img, descriptors1, descriptors2, qs_kp, db_kp, options):
    """ Returns number of matches for descriptor"""
    if type(descriptors1) != np.ndarray or type(descriptors2) != np.ndarray:
        return 0
    
    metric = options.bf_metric 
    max_ratio = options.bf_max_ratio
    matches = match_descriptors(descriptors1, descriptors2, metric=metric, max_ratio=max_ratio, max_distance=0.8, p=1)
    # TODO: Add visualization tool if debug is enabled for point correspondance
    
    if isDebug() and len(matches) > 10:
        show_matches(qs_img, db_img, qs_kp, db_kp, matches)

    return len(matches)

def auto_bf_matcher_cv(db_img, qs_img, descriptors1, descriptors2, qs_kp, db_kp, options):
    """ Returns number of matches for descriptor"""
    if type(descriptors1) != np.ndarray or type(descriptors2) != np.ndarray:
        return 0

    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True) 
    matches = bf.knnMatch(descriptors1,descriptors2,k=1)
    # Apply ratio test
    """ good = []
    for m,n in matches: 
        if m.distance < 0.8*n.distance:
            good.append([m])
    matches = good """
    #if isDebug() and len(matches) > 10:
    #    show_matches(db_img, qs_img, db_kp, qs_kp, matches)

    return len(matches)

def show_matches(img1, img2, kp1, kp2, matches):
    # Need to draw only good matches, so create a mask
    kp1 = np.array([[int(kp.pt[0]), int(kp.pt[1])] for kp in kp1])
    kp2 = np.array([[int(kp.pt[0]), int(kp.pt[1])] for kp in kp2])

    fig, ax = plt.subplots(nrows=2, ncols=1)
    plot_matches(ax[0], cv2.resize(img1, (256,256)), cv2.resize(img2, (256,256)), kp1, kp2, matches)
    ax[0].axis('off')
    ax[0].set_title("Matches")
    plt.show()

    """ matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,m in enumerate(matches):
        if len(matches[i]) != 0:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    
    img3 = cv2.drawMatchesKnn(cv2.resize(img1,(256,256)),kp1,cv2.resize(img2,(256,256)),kp2,matches,None,**draw_params)
    cv2.imshow("matches", img3)
    cv2.waitKey(0) """

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
