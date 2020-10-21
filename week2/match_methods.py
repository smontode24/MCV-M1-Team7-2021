import numpy as np
from metrics import *
import cv2
from debug_utils import *

def painting_matching(imgs, db_imgs, method_name, metric=js_div, splits=30, max_rank=10): 
    """ Obtain query images matches.
        Params:
            - imgs: query set of images [img1, img2,...]
            - db_imgs: database images
            - metric: Similarity measure to quantify the distance between to histograms
            - method_name: method to apply
            - metric: l1_dist,... (which distance / similarity to use)
        Returns: 
            Top k matches for each image of the query set in the database
    """
    matching_method = get_method(method_name)

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
        
        # concatenate all the results
        scores = np.concatenate(scores, 1)
    else:
        db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs])
        scores = metric(query_descriptors, db_descriptors)
    
    top_k_matches = np.argpartition(scores, list(range(max_rank)))[:, :max_rank]
    return top_k_matches

def remove_frame(img):
    """ Remove frame from painting (arbitrarly ~5% of the image on each side) -> Helps in getting better MAP """
    m = 0.05
    p1, p2 = int(img.shape[0]*m), int(img.shape[1]*m)
    img = img[p1:img.shape[0]-p1, p2:img.shape[0]-p2]
    return img, p1, p2

######
# Image descriptors
######

def celled_1dhist(img, cells=[12, 12]):
    """ Divide image into cells and compute the 1d histogram. 
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    for i in range(cells[0]):
        for j in range(cells[1]):
            for ch in range(3):
                vals = np.histogram(img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], ch], bins=np.arange(255))[0]
                normalized_hist = vals/vals.sum()
                descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def celled_2dhist(img, cells=[16, 16]):
    """ Divide image in cells and compute the 2d histogram in another color space.
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    img, p1, p2 = remove_frame(img)

    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    for i in range(cells[0]):
        for j in range(cells[1]):
            cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 1].reshape(-1)
            cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 10), np.arange(0, 255, 10)))[0]
            normalized_hist = vals/vals.sum()
            descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def oned_hist(img):
    """ One dimensional histogram of images. 
        returns: Image descriptor (np.array)
    """
    descriptor = []

    if len(img.shape) == 3:
        img = img.reshape(img.shape[0]*img.shape[1], 3)
    img = img[np.newaxis, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    img_part = img
    cr = img_part[:, :, 1].reshape(-1)
    cb = img_part[:, :, 2].reshape(-1)
    vals = np.histogram(cr, bins=(np.arange(0, 255, 20), np.arange(0, 255, 20)))[0]
    normalized_hist = vals/vals.sum()
    vals2 = np.histogram(img_part[:, :, 1], bins=(np.arange(0, 255, 20)))[0]
    normalized_hist2 = vals2/vals2.sum()
    descriptor.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
    
    return np.array(descriptor).reshape(-1)

def twod_hist(img):
    """ Two dimensional histogram of images. 
        returns: Image descriptor (np.array)
    """
    descriptor = []
    if len(img.shape) == 3:
        img = img.reshape(img.shape[0]*img.shape[1], 3)
    img = img[np.newaxis, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb) 

    img_part = img
    cr = img_part[:, :, 1].reshape(-1)
    cb = img_part[:, :, 2].reshape(-1)
    vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 5), np.arange(0, 255, 5)))[0]
    normalized_hist = vals/vals.sum()
    vals2 = np.histogram(img_part[:, :, 0], bins=(np.arange(0, 255, 5)))[0]
    normalized_hist2 = vals2/vals2.sum()
    descriptor.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
    
    return np.array(descriptor).reshape(-1)

OPTIONS = ["onedcelled", "CBHC", "1dhist", "2dhist"]

METHOD_MAPPING = {
    OPTIONS[0]: celled_1dhist,
    OPTIONS[1]: celled_2dhist,
    OPTIONS[2]: oned_hist,
    OPTIONS[3]: twod_hist
}

def get_method(method):
    return METHOD_MAPPING[method]
