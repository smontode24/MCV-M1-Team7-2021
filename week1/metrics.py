import numpy as np
from scipy.spatial.distance import cdist
import cv2
""" Here create metrics: l2, l1, similarity metrics,... """

DEBUGGING = False  #common flag to know if the main program is requesting debug (as metrics it's used everywhere)
DEBUG_IMAGE = 255 * np.ones(shape=[15, 15, 3], dtype=np.uint8)
DEBUG_IMG_LIST = []

def addDebugImage(img):
    global DEBUG_IMAGE
    global DEBUG_IMG_LIST
    #check size of new image
    height, width, channels = img.shape
    print ('New img was added with H: ' + str(height) + ', W: ' + str(width) + ', Ch: ' + str(channels))
    DEBUG_IMG_LIST.append(img)
    # Check the image that has less width
    w_min = min(im.shape[1] for im in DEBUG_IMG_LIST)
    im_list_resized = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), cv2.INTER_CUBIC)
                      for im in DEBUG_IMG_LIST]
    DEBUG_IMAGE = cv2.vconcat(im_list_resized)
    return DEBUG_IMAGE

def showDebugImage():
    cv2.imshow('Debug Image', DEBUG_IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def setDebugMode(var):
    global DEBUGGING
    DEBUGGING = var
    return DEBUGGING

def isDebug():
    return DEBUGGING

def l2_dist(m1, m2):
    """ m1 (mxd), m2 (nxd) => similarities (mxn) (-distance) """
    x2 = np.sum(m1**2, axis=1, keepdims=True)
    y2 = np.sum(m2**2, axis=1)
    xy = np.dot(m1, m2.T)
    dist = np.sqrt(x2 - 2*xy + y2)
    return dist # Most similar are bigger

def l1_dist(m1, m2):
    """ m1 (mxd), m2 (nxd) => similarities (mxn) (-distance) """
    return cdist(m1, m2, 'minkowski', p=1) # Most similar are bigger

def l1_dist_median(m1, m2):
    total = []
    
    step = m1.shape[1]//100
    for i in range(99):
        total.append(cdist(m1[:,step*i:step*(i+1)], m2[:,step*i:step*(i+1)], 'minkowski', p=1))
    total.append(cdist(m1[:,step*(i+1):], m2[:,step*(i+1):], 'minkowski', p=1))
    total = np.array(total)
    res = np.mean(total, 0)
    return res

def hellinger_kernel(m1, m2):
    """ m1 (mxd), m2 (nxd) => similarities (mxn) (-distance) """
    return -np.sqrt(np.dot(m1, m2.T))

def get_correlation(m1, m2):  
    '''
    Correlation, implemented according to opencv documentation on histogram comparison
    '''
    result = []

    for i in range(m1.shape[0]):
        row = []
        for j in range(m2.shape[0]):
            dev_a = (m1[i,:] - np.mean(m1[i,:]))
            dev_b = (m2[j,:] - np.mean(m2[j,:]))

            row.append(np.sum(dev_a*dev_b) / np.sqrt(np.sum(dev_a*dev_a)*np.sum(dev_b*dev_b)))
        
        result.append(row)
    return -np.array(result)

def js_div(m1, m2):
    result = []

    m1 += 1e-9
    m2 += 1e-9
    
    m1 = m1/m1.sum(axis=1, keepdims=True)
    m2 = m2/m2.sum(axis=1, keepdims=True)
    for i in range(m1.shape[0]):
        for j in range(3): #TODO
            kl_m1_m2 = (m1[i,:]*np.log(m1[i,:]/m2)).sum(axis=1)
            kl_m2_m1 = (m2*np.log(m2/m1[i,:])).sum(axis=1)
            result.append((kl_m1_m2+kl_m2_m1)/2)
    
    return np.array(result)