import numpy as np
from metrics import *
import cv2
from skimage import feature

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

def celled_2dhist_multiresolution(img, cells=[[6,6],[9,9]]):
    """ Divide image in cells and compute the 2d histogram in another color space.
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    img, p1, p2 = remove_frame(img)

    descriptor = []
    w,h = img.shape[:2]
    for cell in cells:
        w_ranges = [(i*w)//cell[0] for i in range(cell[0])]+[-1]
        h_ranges = [(i*h)//cell[1] for i in range(cell[1])]+[-1]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
        for i in range(cell[0]):
            for j in range(cell[1]):
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

def mrhm(img, mask=None, num_blocks=16):
    """ Two dimensional histogram of images. 
        returns: Image descriptor (np.array)
    """
    
    if mask is not None:
        mask = (mask!=0).astype(np.uint8)*255
        
    w,h = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    w_ranges = [(i*w)//num_blocks for i in range(num_blocks)]+[-1]
    h_ranges = [(i*h)//num_blocks for i in range(num_blocks)]+[-1]

    features = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            image_block = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1]]
            if mask is not None:
                mask_block = mask[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1]]
            else:
                mask_block = None

            block_feature = cv2.calcHist([image_block],[0,1,2], mask_block, [2,24,24], [0, 256, 0, 256, 0, 256])
            block_feature = cv2.normalize(block_feature, block_feature)
            features.extend(block_feature.flatten())

    return np.stack(features).flatten()

## TEXTURES:

def LBP(img, mask=None, num_blocks=16):
    """
    This function calculates the LBP descriptor for a given image.

    :param img: image used to calculate the LBP function
    :param num_blocks: number of blocks in which both the height and the width will be divided into
    :param mask: binary mask that will be applied to the image
    :return: the LBP feature array
    """

    descriptor = []
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    height, width = gray_img.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = gray_img[i:i + height_block, j:j + width_block]

            if mask is not None:
                block_mask = mask[i:i + height_block, j:j + width_block]
            else:
                block_mask = None

            block_lbp = np.float32(feature.local_binary_pattern(block, 8, 2, method='default'))

            if mask is not None:
                mask = mask[i:i + height_block, j:j + width_block]

            hist = cv2.calcHist([block_lbp], [0], block_mask, [16], [0, 255])
            cv2.normalize(hist, hist)
            descriptor.extend(hist.reshape(-1))

    return descriptor

def DCT(img, mask=None, num_blocks=16):
    """
    This function calculates the DCT texture descriptor for the given image.
    :param img: image used to calculate the DCT function
    :param num_blocks: number of blocks in which both the height and the width will be divided into
    :param mask: binary mask that will be applied to the image
    :return: the DCT feature array
    """

    descriptor = []
    number_coefficients = 100
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    if mask is not None:
        resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
        resized_image = cv2.bitwise_and(resized_img, resized_img, mask=resized_mask)

    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_img.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = grayscale_img[i:i + height_block, j:j + width_block]

            # Step 1: Calculate the DCT
            block_dct = cv2.dct(np.float32(block)/255.0)

            # Step 2: Zig-Zag scan
            zig_zag_scan = np.concatenate([np.diagonal(block_dct[::-1, :], i)[::(2*(i % 2)-1)]
                                           for i in range(1-block_dct.shape[0], block_dct.shape[0])])

            # Step 3: Keep first N coefficients
            descriptor.extend(zig_zag_scan[:number_coefficients])

    return descriptor


def HOG(img, mask=None):
    """
    Computes the HOG (Histogram of Oriented Gradients) of the given image.
    :param img: image to which the HOG will be calculated
    :param mask: binary mask that will be applied to the image
    :return: array with the image features
    """
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    if grayscale:
        resized_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        multichannel = False

    return feature.hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                       block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True,
                       multichannel=multichannel)


OPTIONS = ["onedcelled", "CBHC", "1dhist", "2dhist", "CBHCM", "LBP", "DCT", "HOG"]

METHOD_MAPPING = {
    OPTIONS[0]: celled_1dhist,
    OPTIONS[1]: celled_2dhist,
    OPTIONS[2]: oned_hist,
    OPTIONS[3]: twod_hist,
    OPTIONS[4]: celled_2dhist_multiresolution,
    OPTIONS[5]: LBP,
    OPTIONS[6]: DCT,
    OPTIONS[7]: HOG
}

def get_method(method):
    return METHOD_MAPPING[method]


#######
TC = "text"
HC = "HC"

OPTIONS = [TC, HC]

METHOD_MAPPING_EXTR = {
    OPTIONS[0]: TC,
    OPTIONS[1]: celled_2dhist_multiresolution
}

def get_descriptor_extractor(method):
    return METHOD_MAPPING_EXTR[method]
