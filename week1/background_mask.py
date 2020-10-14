import numpy as np
import cv2
from metrics import *

""" TODO: Painting region masking:
- Color histograms
- Locate countours (high x and y gradients)
- Morphological operations to fill holes / remove unnecesary isolated parts
- Histogram in cells, and compare to adjacent ones. If different->change bg/fg.
"""

def bg_mask(query_imgs, method): 
    """ Obtain mask from image """
    print("Obtaining masks")
    segmentation_method = get_method(method)
    return [segmentation_method(img) for img in query_imgs]

def apply_mask(query_imgs, masks, method): 
    """ Apply mask to each image in the query set, based on the method that was applied """
    resulting_imgs = []
    for img, mask in zip(query_imgs, masks):
        positions = np.where(mask == 255)
        if method == CBHS:
            x_min, x_max, y_min, y_max = positions[0][0], positions[0][-1], positions[1][0], positions[1][-1]
            img = img[x_min:x_max, y_min:y_max]
        elif method == GRADIENT:
            print('STUFF PENDING TO DO')
        else:
            if isDebug():
                
                mask_res = np.zeros((img.shape[0], img.shape[1], 3))
                for i in range(3):
                    mask_res[:,:,i] = mask

                cv2.imshow("masked", mask_res)
                cv2.waitKey(0)

            mask = mask == 255
            img = img[mask]
        resulting_imgs.append(img)
        if isDebug():
            addDebugImage(img)
    if isDebug():
        showDebugImage()
        print("Finished to apply masks")
    return resulting_imgs

def grad(img, cells=[32, 32]):
    print("I'm here!!")

def cell_based_hist_segmentation(img, cells=[32, 32]):
    """ Create a mask from image that indicates where the painting is located in the image """
    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    celled_hist = obtain_celled_histograms(img, cells, w_ranges, h_ranges)

    rectangular_segm = obtain_rectangular_segmentation(celled_hist, cells)
    mask = create_mask(rectangular_segm, img, cells)
    if isDebug():
        cv2.imshow("original", img)
        cv2.waitKey(1)
        positions = np.where(mask == 255)
        l,r,t,b = positions[0][0], positions[0][-1], positions[1][0], positions[1][-1]
        masked_img = img[l:r, t:b]
        cv2.imshow("masked", masked_img)
        cv2.waitKey(0)
    return mask

def create_mask(masking_matrix, img, cells):
    """ Create mask from the positions that were estimated in the celled based histogram method """
    left, right, top, bottom = masking_matrix
    left += 1
    right += 1
    top += 1
    bottom += 1
    mask = np.ones((img.shape[0], img.shape[1]))*255

    # Compute corresponding positions and put zeros in the background part
    left = (img.shape[1]//cells[0])*left
    mask[:, :left] = 0
    right = img.shape[1]-(img.shape[1]//cells[0])*right
    mask[:, right:] = 0
    top = (img.shape[0]//cells[1])*top
    mask[:top, :] = 0
    bottom = img.shape[0]-(img.shape[0]//cells[0])*bottom
    mask[bottom:, :] = 0

    masks = mask.astype(np.uint8)
    return mask

def histogram_segmentation(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) 
    
    # Segmentation based on border statistics
    l_mean = (np.mean(img[0, :, 0]) + np.mean(img[:, 0, 0]) + np.mean(img[-1, :, 0]) + np.mean(img[:, -1, 0]))/4
    a_mean = (np.mean(img[0, :, 1]) + np.mean(img[:, 0, 1]) + np.mean(img[-1, :, 1]) + np.mean(img[:, -1, 1]))/4
    b_mean = (np.mean(img[0, :, 2]) + np.mean(img[:, 0, 2]) + np.mean(img[-1, :, 2]) + np.mean(img[:, -1, 2]))/4

    l_std = (np.concatenate([img[0, :, 0].reshape(-1), img[:, 0, 0].reshape(-1), img[-1, :, 0].reshape(-1), img[:, -1, 0].reshape(-1)])).std()
    a_std = (np.concatenate([img[0, :, 1].reshape(-1), img[:, 0, 1].reshape(-1), img[-1, :, 1].reshape(-1), img[:, -1, 1].reshape(-1)])).std()
    b_std = (np.concatenate([img[0, :, 2].reshape(-1), img[:, 0, 2].reshape(-1), img[-1, :, 2].reshape(-1), img[:, -1, 2].reshape(-1)])).std()

    ts = 3
    lower_limit = np.array([l_mean-ts*l_std, a_mean-ts*a_std, b_mean-ts*b_std])
    upper_limit = np.array([l_mean+ts*l_std, a_mean+ts*a_std, b_mean+ts*b_std])

    # Create mask
    mask = (cv2.inRange(img, lower_limit.astype(np.int), upper_limit.astype(np.int)) != 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    new_mask = cv2.fillPoly(mask, contours, 255)
    return new_mask

def sm(x):
    """ Softmax """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def compute_positions(scores, layers, cells, direction):
    """ Compute position that has greater change in the histogram """
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
    position = np.argmax(col_scores*prior)
    return position

def obtain_rectangular_segmentation(celled_hist, cells):
    """ Obtain left, top, right and bottom positions in the celled histogram that provides maximum change """
    celled_hist = np.array(celled_hist).reshape(cells[0]*cells[1], -1)
    scores = l1_dist(celled_hist, celled_hist)

    layers = cells[0]//2
    
    left, right, top, bottom = compute_positions(scores, layers, cells, "lr"), compute_positions(scores, layers, cells, "rl"), \
                               compute_positions(scores, layers, cells, "tb"), compute_positions(scores, layers, cells, "bt")

    return left, right, top, bottom

def obtain_celled_histograms(img, cells, w_ranges, h_ranges):
    """ Rectangular mask segmentation. It compares the change in histograms for each consecutive row/column to create
    a rectangular region. """
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
            row.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
        histogram_matrix.append(row)
    return histogram_matrix

def dummy(img):
    return np.ones((img.shape[0], img.shape[1])) == 1

METHODS = ["cbhs", "grad", "hist"]
CBHS = "cbhs"
GRADIENT = "grad"

MAP_METHOD = {
    METHODS[0]: cell_based_hist_segmentation,
    METHODS[1]: grad,
    METHODS[2]: histogram_segmentation
}

def get_method(method=1):
    return MAP_METHOD[method]