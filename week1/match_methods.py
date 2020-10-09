import numpy as np
from metrics import *
import cv2

""" TODO: Matching methods:
- Color histograms
- ...
"""

def painting_matching(imgs, db_imgs, splits=30, max_rank=5): 
    """ Obtain from image """
    matching_method = get_method(2)
    metric = l1_dist

    db_img_splits = [i*len(db_imgs)//splits for i in range(splits-1)]
    
    scores = []
    query_descriptors = np.array([matching_method(img) for img in imgs])
    print("Starting db extraction + matching")
    if splits > 1:
        for split in range(splits-2):
            db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs[db_img_splits[split]:db_img_splits[split+1]]])
            scores.append(metric(query_descriptors, db_descriptors))
        
        db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs[db_img_splits[-1]:]])
        scores.append(metric(query_descriptors, db_descriptors))
        # concatenate
        scores = np.concatenate(scores, 1)
    else:
        db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs])
        scores = metric(query_descriptors, db_descriptors)
    
    top_k_matches = np.argpartition(scores, list(range(max_rank)))[:, :max_rank]
    return top_k_matches

def celled_hist(img, cells=[12, 12]):
    """ Divide image in cells and compute the histogram.
    Downsides: bigger descriptor, rotation, illumination (?) """
    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

    """ for ch in range(1,3):
        vals = np.histogram(img[:, :, ch], bins=np.arange(255))[0]
        normalized_hist = vals/vals.sum()
        descriptor.append(normalized_hist) """

    for i in range(cells[0]):
        for j in range(cells[1]):
            for ch in range(3):
                vals = np.histogram(img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], ch], bins=np.arange(255))[0]
                normalized_hist = vals/vals.sum()
                descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def dummy(img):
    return img.reshape(-1)

def celled_2dhist(img, cells=[32, 32]): # opt: 14
    """ Divide image in cells and compute the histogram.
    Downsides: bigger descriptor, rotation, illumination (?) """

    m = 0.05
    p1, p2 = int(img.shape[0]*m), int(img.shape[1]*m)
    img = img[p1:img.shape[0]-p1, p2:img.shape[0]-p2]

    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    for i in range(cells[0]):
        for j in range(cells[1]):
            cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], 1].reshape(-1)
            cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(42, 226, 20), np.arange(20, 223, 20)))[0]
            normalized_hist = vals/vals.sum()
            descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def celled_2dhist2(img, cells=[12, 12]):
    """ Divide image in cells and compute the histogram.
    Downsides: bigger descriptor, rotation, illumination (?) """
    m = 0.05
    p1, p2 = int(img.shape[0]*m), int(img.shape[1]*m)
    img = img[p1:img.shape[0]-p1, p2:img.shape[0]-p2]

    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for i in range(cells[0]):
        for j in range(cells[1]):
            cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], 0].reshape(-1)
            cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 5), np.arange(0, 255, 5)))[0]
            normalized_hist = vals/vals.sum()
            descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)


METHOD_MAPPING = {
    1: celled_hist,
    2: celled_2dhist,
    3: celled_2dhist2,
    4: dummy
}

def get_method(method):
    return METHOD_MAPPING[method]