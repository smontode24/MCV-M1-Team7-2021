import numpy as np
from tqdm import tqdm
from metrics import *

""" TODO: Matching methods:
- Color histograms
- ...
"""

def painting_matching(imgs, db_imgs, max_rank=5): 
    """ Obtain from image """
    print("Obtaining query descriptors")
    query_descriptors = np.array([celled_hist(img) for img in imgs])
    print("Obtaining DB descriptors")
    db_descriptors = np.array([celled_hist(db_img) for db_img in db_imgs])
    print("Computing scores")
    scores = l2_dist(query_descriptors, db_descriptors)
    top_k_matches = np.argpartition(scores, list(range(max_rank)))[:, :max_rank]
    return top_k_matches

def celled_hist(img, cells=[5,5]):
    """ Divide image in cells and compute the histogram.
    Downsides: bigger descriptor, rotation, illumination (?) """
    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    """ for ch in range(3):
        vals = np.histogram(img[:, :, ch], bins=np.arange(0, 255, 5))[0]
        normalized_hist = vals/vals.sum()
        descriptor.append(normalized_hist) """

    for i in range(cells[0]):
        for j in range(cells[1]):
            for ch in range(3):
                vals = np.histogram(img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], ch], bins=np.arange(255))[0]
                normalized_hist = vals/vals.sum()
                descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)