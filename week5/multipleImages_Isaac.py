import argparse
from io_utils import *
from io_utils import *
from debug_utils import *
import numpy as np
import imutils
from skimage.io import imread_collection
from descriptors import HOG

# Own coded comparator, in order to order nested lists in Python
def compareDistances (dMatch):
    return dMatch.distance

# RETURN: Given a relative path, it return it's absolute
def absolutePath(relative):
    # Join of a first system separator + PWD function + Relative = Abosulte path
    return os.path.join(os.path.sep, os.path.dirname(__file__), relative)

# RETURN: An image specified on the path (absolute or relative)
def openImage(path):
    img = cv2.imread(path)

    return img

# IMG = 3 channel RGB image (ndarray)
# RETURN: An image with 1 dimension, gray
def rgb2gray(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_image

# RETURN: An image with its gradient in X dimension
def gradientX (img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)

# RETURN: An image with its gradient in Y dimension
def gradientY (img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Credits: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections

def HOG_extractor(image_path, features):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=1024)
    hog = HOG(image, None)
    tuple = [image_path, hog]
    features.append(tuple)

def k_means(image_path):
    # following the previous guide +++ :
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#kmeans-opencv
    # DOCUMENTATION: https://docs.opencv.org/master/d5/d38/group__core__cluster.html

    image = cv2.imread(image_path)
    image = imutils.resize(image, width=1024)
    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # TODO: El noi de pyImageSearch fa una barra de codis
    # reshape the image to be a list of pixels
    # oneDarray = image.reshape((image.shape[0] * image.shape[1], 3))
    # clt = cv2.kmeans(K=options.clusters)
    # clt.fit(image)
    # End of TODO

    ret, label, center = cv2.kmeans(Z, 32, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    #cv2.imshow('res2', res2)

    return res2



def do_pipeline_RGB(path, mser):
    I = openImage(path)
    gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    I = imutils.resize(I, width=1024)
    gray = imutils.resize(gray, width=1024)
    #gray = cv2.resize(gray,(500,400),interpolation=cv2.INTER_LANCZOS4)
    #ret, threshold = cv2.threshold(gray, 40, 255, cv2.THRESH_TOZERO)
    msers, bboxes = mser.detectRegions(gray)
    i = 0
    # DETAIL
    # Avoid very smalll boxes
    # Avoid images with y-component > 20%
    big_areas=[]
    for p in msers:
        try:
            xmax, ymax = np.amax(p, axis=0)
            xmin, ymin = np.amin(p, axis=0)
            # Check size between points
            # just print those bigs!!
            if 5 < (xmax-xmin) < 0.3*width:
                if 3 < (ymax-ymin) < 0.2*height:
                    #print ("   it has rectangle: "+str(i))
                    big_areas.append(p)
        except:
            print("   "+str(i)+"   produced oops")
    percentage = 0.01       #Consider a deviation of 1% of the total height
    y_base_desviation = percentage*height
    desviation_y = height*percentage
    # processing horitzontal msers in bbxes that have a "big area" [enough from previous filters]
    horitzontal_blocks = []
    for index1, j in enumerate(big_areas):
        i = 0
        xmax_orig, ymax_orig = np.amax(j, axis=0)
        xmin_orig, ymin_orig = np.amin(j, axis=0)
        xmax_final=0
        xmin_final=0
        ymax_final=0
        ymin_final=0
        del big_areas[index1]
        for index2, q in enumerate(big_areas):
            xmax_cand, ymax_cand = np.amax(q, axis=0)
            xmin_cand, ymin_cand = np.amin(q, axis=0)
            # Checking, y_start point is not desviate > percentage
            # Fine Tunning: Required to be also up & down... because i letters (and those that go down like: q, j, g)
            if  (ymin_orig-0.6*(ymax_orig-ymin_orig) < ymin_cand < ymin_orig+0.6*(ymax_orig-ymin_orig)) and \
                (ymax_orig-0.6*(ymax_orig-ymin_orig) < ymax_cand < ymax_orig+0.6*(ymax_orig-ymin_orig)):
                    # It has to have a neighbour, also next to it....
                    # Represented by: starting point (xmin_cand) cannot be after more than 3 * space of the
                    # previous letter size
                    if  (xmax_orig < xmin_cand < xmax_orig+3*(xmax_orig-xmin_orig)):
                        horitzontal_blocks.append(q)
                        cv2.rectangle(I, (xmin_orig, ymax_orig), (xmax_orig, ymin_orig), (0, 255, 0), 3)
                        cv2.rectangle(I, (xmin_cand, ymax_cand), (xmax_cand, ymin_cand), (0, 0, 255), 1)
                        del big_areas[index2]
                        i = i+1

    return I


# Reusing some calls from the MAIN code,
# this program focused in one imag
if __name__ == "__main__":
    # Path to DB image:
    path0 = "BBDD/*.jpg"
    path0 = absolutePath(path0)

    # Path to image in DB
    path1 = "qsd1_w5/*.jpg"
    path1 = absolutePath(path1)
    print("Image from the qds1_w5: ", path1)

    # creating a collection with the available images
    images = imread_collection(path0).files

    i = 0
    path2 = "output_mask/"
    path2 = absolutePath(path2)
    # Features vector
    HOG_features = []

    for img in images:
        #k_means(img)
        #HOG_extractor(img, HOG_features)

        result = k_means(img)
        basename = img.rpartition('/')[-1]  # gets last filename from
        path3 = path2+basename
        print(path3)
        cv2.imwrite(path3, result)
        i=i+1
    print("hey, stop!")