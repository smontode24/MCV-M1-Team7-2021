import numpy as np
import cv2
from metrics import *
import scipy.stats as stats
from debug_utils import *

def bg_mask(query_imgs, method): 
    """ Obtain a mask for each image in the list of images query_img. The method will be determined by the passed method argument.
    Available methods are: "CBHS" and "PBM". 
        params: 
            query_imgs: List of images of the query set
            method: ["CBHS", "PBM"]
        returns:
            List of masks [2D images with 1 channel]. A pixel = 0 means background, and pixel = 255 painting
    """
    print("Obtaining masks")
    segmentation_method = get_method(method)
    return [segmentation_method(img) for img in query_imgs]

def apply_mask(query_imgs, masks, method): 
    """ Apply mask to each image in the query set, based on the method that was applied 
            params:
                query_imgs: List of images that will be masked
                masks: List of masks to apply to each image
        returns: 
                List of masked images
    """
    resulting_imgs = []
    for img, mask in zip(query_imgs, masks):
        positions = np.where(mask == 255)
        if method == CBHS: # Special treatment for cell-based bg segmentation to mantain 
            x_min, x_max, y_min, y_max = positions[0][0], positions[0][-1], positions[1][0], positions[1][-1]
            img = img[x_min:x_max, y_min:y_max]
        else:
            mask = mask == 255
            img = img[mask].reshape(-1, 3)
        resulting_imgs.append(img)
        
        if isDebug():
            addDebugImage(img)
    if isDebug():
        showDebugImage()
        print("Finished to apply masks")
    
    return resulting_imgs

def cell_based_hist_segmentation(img, cells=[32, 32]):
    """ Create a mask from image that indicates where the painting is located in the image.
            params:
                img: image where the mask will be estimated
                cells: list with two elements indicating the amount of cells in which the image will be decomposed. 
                        e.g., cells=[32,32] divide image into 32x32 cells.
            returns:
                Mask indicating the part that corresponds to the painting 
    """
    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    celled_hist = obtain_celled_histograms(img, cells, w_ranges, h_ranges)

    rectangular_segm = obtain_rectangular_segmentation(celled_hist, cells)
    mask = create_mask(rectangular_segm, img, cells)
    return mask

def create_mask(masking_positions, img, cells):
    """ Create mask from the positions that were estimated in the celled-based histogram method.
        params:
            masking_positions: positions where the rectangle positions have been detected (left, right, top, bottom). These positions refer to positions
            in the image that was decomposed into cells.
            img: Image where the mask will be applied.
            cells: Cells that the image has been decomposed to. (e.g., [32,32])
        returns:
            2D mask with 1 channel. 0 = background, 255 = painting
     """
    left, right, top, bottom = masking_positions
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

def pbm_segmentation(img, margin=0.02, threshold=0.00001):
    """ Probability-based segmentation. Model the background with a multivariate gaussian distribution. 
            img: Image to be segmented
            margin: Part of image being certainly part of the background 
            threshold: Threshold to reject a pixel as not being part of the background
        returns: Mask image, indicating the painting part in the image
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) 
    h_m, w_m = int(img.shape[0]*margin), int(img.shape[1]*margin)

    # Compute mean and standard deviation for each channel separately
    l_mean = (np.concatenate([img[:h_m, :, 0].reshape(-1), img[:, :w_m, 0].reshape(-1), img[img.shape[0]-h_m:, :, 0].reshape(-1), \
                             img[:, img.shape[1]-w_m:, 0].reshape(-1)])).mean()
    a_mean = (np.concatenate([img[:h_m, :, 1].reshape(-1), img[:, :w_m, 1].reshape(-1), img[img.shape[0]-h_m:, :, 1].reshape(-1), \
                             img[:, img.shape[1]-w_m:, 1].reshape(-1)])).mean()
    b_mean = (np.concatenate([img[:h_m, :, 2].reshape(-1), img[:, :w_m, 2].reshape(-1), img[img.shape[0]-h_m:, :, 2].reshape(-1), \
                             img[:, img.shape[1]-w_m:, 2].reshape(-1)])).mean()

    l_std = (np.concatenate([img[:h_m, :, 0].reshape(-1), img[:, :w_m, 0].reshape(-1), img[img.shape[0]-h_m:, :, 0].reshape(-1), \
                             img[:, img.shape[1]-w_m:, 0].reshape(-1)])).std()
    a_std = (np.concatenate([img[:h_m, :, 1].reshape(-1), img[:, :w_m, 1].reshape(-1), img[img.shape[0]-h_m:, :, 1].reshape(-1), \
                             img[:, img.shape[1]-w_m:, 1].reshape(-1)])).std()
    b_std = (np.concatenate([img[:h_m, :, 2].reshape(-1), img[:, :w_m, 2].reshape(-1), img[img.shape[0]-h_m:, :, 2].reshape(-1), \
                             img[:, img.shape[1]-w_m:, 2].reshape(-1)])).std()

    # Model background and discard unlikely pixels
    mask = stats.norm.pdf(img[:,:,0], l_mean, l_std)*stats.norm.pdf(img[:,:,1], a_mean, a_std)*stats.norm.pdf(img[:,:,2], b_mean, b_std) < threshold
    
    new_mask = np.zeros_like(mask).astype(np.uint8)
    new_mask[mask] = 255
    
    return new_mask

def sm(x):
    """ Softmax """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def compute_positions(scores, layers, cells, direction):
    """ Compute position that has greater change in the cell histogram matrix.
        params:
            scores: (c0xc1)x(c0xc1) matrix that contains the similarity between the cell k and cell j in the position [k,j] (flattened cell grid).
            layers: number of layers until reaching the center
            direction: lr, rl, tb or bt
        returns: Position of maximum change
     """
    prior = 1/np.arange(3, layers+3)
    prior = prior/prior.sum()
    x = np.linspace(-5, 5, layers)[::-1] 
    prior = 1/(1 + np.exp(-x))
    prior = prior/prior.sum()

    # Compute the probability depending on the direction in which we want to know where the change is
    if direction == "lr":
        col_scores = np.array([scores[np.arange(cells[0])*cells[1]+j, np.arange(cells[0])*cells[1]+j+1].sum() for j in range(layers)])
    elif direction == "rl":
        col_scores = np.array([scores[np.arange(cells[0])*cells[1]+cells[1]-1-j, np.arange(cells[0])*cells[1]+cells[1]-1-j-1].sum() for j in range(layers)])
    elif direction == "tb":
        col_scores = np.array([scores[np.arange(cells[1])+cells[1]*j, np.arange(cells[1])+cells[1]*(j+1)].sum() for j in range(layers)])
    elif direction == "bt":
        col_scores = np.array([scores[np.arange(cells[1])+cells[1]*(cells[0]-1-j), np.arange(cells[1])+cells[1]*(cells[0]-1-j-1)].sum() for j in range(layers)])
     
    # Apply softmax + multiply by prior -> Then get the most likely position
    col_scores = sm(col_scores)
    position = np.argmax(col_scores*prior)
    return position

def obtain_rectangular_segmentation(celled_hist, cells):
    """ Obtain left, top, right and bottom positions in the celled histogram that provides maximum change. With these positions
    a rectangular matrix can be constructed.
    params:
        celled_hist: matrix (size depends on the amount of cells) with 2D histograms (one for each cell)
        cells: Size of the cell division (e.g., cells=[32,32] divides image into 32x32 cells)
    returns: Rectangle coordinates for the rectangular segmentation [left, right, top, bottom]
    """
    celled_hist = np.array(celled_hist).reshape(cells[0]*cells[1], -1)
    scores = l1_dist(celled_hist, celled_hist)

    layers = cells[0]//2
    
    left, right, top, bottom = compute_positions(scores, layers, cells, "lr"), compute_positions(scores, layers, cells, "rl"), \
                               compute_positions(scores, layers, cells, "tb"), compute_positions(scores, layers, cells, "bt")

    return left, right, top, bottom

def obtain_celled_histograms(img, cells, w_ranges, h_ranges):
    """ Compute a cell matrix histogram. If cells=[32,32], returns a 32x32 matrix where each position contains a 2D color histogram (image descriptor).
        params:
            img: Image that will have the descriptor extracted 
            cells: Cells that the image has been decomposed to. (e.g., [32,32])
            w_ranges and h_ranges indicate the pixel position divisions of the cells.
            Example:
                cell = [4,4]
                img shape = [48,48]
                w_ranges = [0, 12, 24, 36, 48]
                h_ranges = [0, 12, 24, 36, 48]
        returns:
            Image descriptor
    """
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
            vals2 = np.histogram(img_part[:, :, 0], bins=(np.arange(0, 255, 20)))[0]
            normalized_hist2 = vals2/vals2.sum()
            row.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
        histogram_matrix.append(row)
    return histogram_matrix

CBHS = "CBHS"
PBM = "PBM"
METHODS = [CBHS, PBM]

MAP_METHOD = {
    METHODS[0]: cell_based_hist_segmentation,
    METHODS[1]: pbm_segmentation
}

def get_method(method=1):
    return MAP_METHOD[method]