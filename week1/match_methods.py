import numpy as np
from metrics import *
import cv2

""" TODO: Matching methods:
- Color histograms
- Color histogram with gradient images
- FFT? Remove hf, sparkles....
"""

def painting_matching(imgs, db_imgs, splits=30, max_rank=5): 
    """ Obtain from image """
    matching_method = get_method(2)
    metric = get_correlation

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

def celled_2dhist(img, cells=[16, 16]): # opt: 14
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
            cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 1].reshape(-1)
            cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(42, 226, 10), np.arange(20, 223, 10)))[0]
            #vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 20), np.arange(0, 255, 20)))[0]
            normalized_hist = vals/vals.sum()
            descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def celled_2dhist2(img, cells=[16, 16]): # opt: 14
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
            img_part = img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1]]
            cr = img_part[:, :, 1].reshape(-1)
            cb = img_part[:, :, 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 10), np.arange(0, 255, 10)))[0]
            normalized_hist = vals/vals.sum()
            #vals2 = np.histogram(img_part[:, :, 1], bins=(np.arange(0, 255, 20)))[0]
            #normalized_hist2 = vals2/vals2.sum()
            descriptor.append(normalized_hist)
            #descriptor.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
    
    return np.array(descriptor).reshape(-1)

def celled_2dhist3(img): # opt: 14
    """ Divide image in cells and compute the histogram.
    Downsides: bigger descriptor, rotation, illumination (?) """

    m = 0.05
    p1, p2 = int(img.shape[0]*m), int(img.shape[1]*m)
    img = img[p1:img.shape[0]-p1, p2:img.shape[0]-p2]

    descriptor = []

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    img_part = img
    cr = img_part[:, :, 1].reshape(-1)
    cb = img_part[:, :, 2].reshape(-1)
    vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 20), np.arange(0, 255, 20)))[0]
    normalized_hist = vals/vals.sum()
    vals2 = np.histogram(img_part[:, :, 1], bins=(np.arange(0, 255, 20)))[0]
    normalized_hist2 = vals2/vals2.sum()
    #descriptor.append(normalized_hist)
    descriptor.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
    
    return np.array(descriptor).reshape(-1)


METHOD_MAPPING = {
    1: celled_hist,
    2: celled_2dhist,
    3: celled_2dhist2,
    4: celled_2dhist3
}

def get_method(method):
    return METHOD_MAPPING[method]