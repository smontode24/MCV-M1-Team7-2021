import numpy as np
import cv2

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