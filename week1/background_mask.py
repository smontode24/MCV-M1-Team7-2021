import numpy as np
import cv2
from metrics import *

""" TODO: Painting region masking:
- Color histograms
- Locate countours (high x and y gradients)
- Morphological operations to fill holes / remove unnecesary isolated parts
- Histogram in cells, and compare to adjacent ones. If different->change bg/fg.
"""

def bg_mask(query_imgs): 
    """ Obtain mask from image """
    # Temporary mask (grab all the painting part)
    print("Obtaining masks")
    segmentation_method = get_method(1)
    return [segmentation_method(img) for img in query_imgs]

def apply_mask(query_imgs, masks): 
    """ Obtain mask from image """
    # Temporary mask (grab all the painting part)
    resulting_imgs = []
    for img, mask in zip(query_imgs, masks):
        positions = np.where(mask == True)
        x_min, x_max, y_min, y_max = positions[0][0], positions[0][-1], positions[1][0], positions[1][-1]
        img = img[x_min:x_max, y_min:y_max] #img[mask==1].reshape(x_max-x_min, y_max-y_min, 3)
        resulting_imgs.append(img)
    return resulting_imgs

def grad_based_segmentation(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    res = np.abs(sobelx)+np.abs(sobely)
    # I think that we cannot do this yet

def cell_based_hist_segmentation(img, cells=[32, 32]):

    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    celled_hist = obtain_celled_histograms(img, cells, w_ranges, h_ranges)

    """ cv2.imshow("pre-painting", img)
    cv2.waitKey(1) """
    rectangular_segm = obtain_rectangular_segmentation(celled_hist, cells)
    mask = create_mask(rectangular_segm, img, cells)
    """for i in range(3):
        img[mask==0] = 0
    cv2.imshow("painting", img)
    cv2.waitKey(0) """
    
    """mask_cp = mask.copy()
    mask_cp = mask_cp*255
    mask_show = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(3):
        mask_show[:,:,i] = mask_cp
    cv2.imshow("mask", mask_show)
    cv2.waitKey(0) """
    return mask

def create_mask(masking_matrix, img, cells):
    left, right, top, bottom = masking_matrix
    left += 1
    right += 1
    top += 1
    bottom += 1
    mask = np.ones((img.shape[0], img.shape[1]))

    left = (img.shape[0]//cells[0])*left
    mask[:, :left] = 0
    right = img.shape[0]-(img.shape[0]//cells[0])*right
    mask[:, right:] = 0
    top = (img.shape[1]//cells[1])*top
    mask[:top, :] = 0
    bottom = img.shape[0]-(img.shape[0]//cells[0])*bottom
    mask[bottom:, :] = 0

    return mask

def sm(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def compute_positions(scores, layers, cells, direction):

    prior = 1/np.arange(3, layers+3)
    prior = prior/prior.sum()
    x = np.linspace(-5, 5, layers)[::-1] 
    prior = 1/(1 + np.exp(-x))
    prior = prior/prior.sum()

    posterior = []

    if direction == "lr":
        col_scores = np.array([scores[np.arange(cells[0])*cells[1]+j, np.arange(cells[0])*cells[1]+j+1].sum() for j in range(layers)])
    elif direction == "rl":
        col_scores = np.array([scores[np.arange(cells[0])*cells[1]+cells[1]-1-j, np.arange(cells[0])*cells[1]+cells[1]-1-j-1].sum() for j in range(layers)])
    elif direction == "tb":
        col_scores = np.array([scores[np.arange(cells[1])+cells[1]*j, np.arange(cells[1])+cells[1]*(j+1)].sum() for j in range(layers)])
    elif direction == "bt":
        col_scores = np.array([scores[np.arange(cells[1])+cells[1]*(cells[0]-1-j), np.arange(cells[1])+cells[1]*(cells[0]-1-j-1)].sum() for j in range(layers)])
     
    col_scores = sm(col_scores)
    position = np.argmax(col_scores*prior) # prior*col_scores
    return position

def obtain_rectangular_segmentation(celled_hist, cells):
    celled_hist = np.array(celled_hist).reshape(cells[0]*cells[1], -1) #.reshape(cells[0], cells[1], -1) #. TODO: check this
    scores = l1_dist(celled_hist, celled_hist)
    #scores = celled_hist@celled_hist.T # ((mxm)xd)@(dx(mxm)) = ((mxm)x(mxm))

    layers = cells[0]//2
    
    # Left->Right
    left, right, top, bottom = compute_positions(scores, layers, cells, "lr"), compute_positions(scores, layers, cells, "rl"), \
                               compute_positions(scores, layers, cells, "tb"), compute_positions(scores, layers, cells, "bt")

    return left, right, top, bottom

def obtain_celled_histograms(img, cells, w_ranges, h_ranges):
    results = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    histogram_matrix = []

    for i in range(cells[0]):
        row = []
        for j in range(cells[1]):
            img_part = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1]]
            cr = img_part[:, :, 1].reshape(-1)
            cb = img_part[:, :, 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(42, 226, 10), np.arange(20, 223, 10)))[0]
            normalized_hist = vals/vals.sum()
            vals2 = np.histogram(img_part[:, :, 1], bins=(np.arange(0, 255, 10)))[0]
            normalized_hist2 = vals2/vals2.sum()
            #row.append(normalized_hist.reshape(-1))
            row.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
        histogram_matrix.append(row)
    return histogram_matrix

MAP_METHOD = {
    1: cell_based_hist_segmentation,
    2: grad_based_segmentation
}

def get_method(method=1):
    return MAP_METHOD[method]