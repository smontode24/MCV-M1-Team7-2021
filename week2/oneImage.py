import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from text_segmentation import *
from metrics import *
from evaluation.mask_evaluation import *
from evaluation.retrieval_evaluation import *
from time import time
from io_utils import *
from debug_utils import *
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
import math


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


def rgb2gray2(img):
    gray_image = img.dot([0.07, 0.72, 0.21])

    return gray_image

# RETURN: An image with its gradient in X dimension
def gradientX (img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)

# RETURN: An image with its gradient in Y dimension
def gradientY (img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)

# Veure gradients x i y, y obtenir els dos rectangles més grans


# Reusing some calls from the MAIN code,
# this program focused in one imag
if __name__ == "__main__":
    # Path to DB image:
    path0 = "BBDD/bbdd_00219.jpg"
    path0 = absolutePath(path0)
    print ("Image from the DB: ", path0)
    # Path to image in QDS1
    path1 = "qsd1_w2/00004.jpg"
    path1 = absolutePath(path1)
    # Path to image in QDS2
    path2 = "qsd2_w2/00000.jpg"
    path2 = absolutePath(path2)
    # Path to mask in QSD2
    path3 = "qsd2_w2/00000.png"
    path3 = absolutePath(path3)

    img0 = openImage(path0)
    img1 = openImage(path1)
    img2 = openImage(path2)
    msk2 = openImage(path3)

    blurred_img2 = cv2.GaussianBlur(img2, (5,5), 0)
    blurred_img2_7 = cv2.GaussianBlur(img2, (7,7), 0)
    blurred_img2_9 = cv2.GaussianBlur(img2, (9,9), 0)
    blurred_img2_11 = cv2.GaussianBlur(img2, (11,11), 0)
    blurred_img2_75 = cv2.GaussianBlur(img2, (75,75), 0)

    #Extracted from here: https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python
    gradX_0 = ndimage.sobel(img0, axis=0, mode='constant')
    gradY_0 = ndimage.sobel(img0, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_0 = np.hypot(gradX_0, gradY_0)

    gradX_1 = ndimage.sobel(img1, axis=0, mode='constant')
    gradY_1 = ndimage.sobel(img1, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_1 = np.hypot(gradX_1, gradY_1)

    gradX_2 = ndimage.sobel(img2, axis=0, mode='constant')
    gradY_2 = ndimage.sobel(img2, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_2 = np.hypot(gradX_2, gradY_2)

    gradX_m2 = ndimage.sobel(msk2, axis=0, mode='constant')
    gradY_m2 = ndimage.sobel(msk2, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_m2 = np.hypot(gradX_m2, gradY_m2)

    ## IMAGES IN BW
    img0_bw = rgb2gray(img0)
    img1_bw = rgb2gray(img1)
    img2_bw = rgb2gray(img2)
    msk2_bw = rgb2gray(msk2)
    blurred_img2_bw = rgb2gray(blurred_img2)
    blurred_img2_7_bw = rgb2gray(blurred_img2_7)
    blurred_img2_9_bw = rgb2gray(blurred_img2_9)
    blurred_img2_11_bw = rgb2gray(blurred_img2_11)
    blurred_img2_75_bw = rgb2gray(blurred_img2_75)

    '''
    ## AVOID TO USE THEM:
    # Why? Because they're very specific. CV ones are much appropiate.
    ## GRADIENTS IN BW  [USING NUMPY]
    gradX_0_bw = ndimage.sobel(img0_bw, axis=0, mode='constant')
    gradY_0_bw = ndimage.sobel(img0_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_0_bw = np.hypot(gradX_0_bw, gradY_0_bw)

    gradX_1_bw = ndimage.sobel(img1_bw, axis=0, mode='constant')
    gradY_1_bw = ndimage.sobel(img1_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_1_bw = np.hypot(gradX_1_bw, gradY_1_bw)

    gradX_2_bw = ndimage.sobel(img2_bw, axis=0, mode='constant')
    gradY_2_bw = ndimage.sobel(img2_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_2_bw = np.hypot(gradX_2_bw, gradY_2_bw)

    gradX_m2_bw = ndimage.sobel(msk2_bw, axis=0, mode='constant')
    gradY_m2_bw = ndimage.sobel(msk2_bw, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel_m2_bw = np.hypot(gradX_m2_bw, gradY_m2_bw)
    '''

    ## These gradients are much better. More general (not tinny detailed)
    ## GRADIENTS IN BW  [USING CV]
    cv_gradX_0_bw = gradientX(img0_bw)
    cv_gradY_0_bw = gradientY(img0_bw)
    # Get square root of sum of squares
    cv_sobel_0_bw = np.hypot(cv_gradX_0_bw, cv_gradY_0_bw)

    cv_gradX_1_bw = gradientX(img1_bw)
    cv_gradY_1_bw = gradientY(img1_bw)
    # Get square root of sum of squares
    cv_sobel_1_bw = np.hypot(cv_gradX_1_bw, cv_gradY_1_bw)

    cv_gradX_2_bw = gradientX(img2_bw)
    cv_gradY_2_bw = gradientY(img2_bw)
    # Get square root of sum of squares
    cv_sobel_2_bw = np.hypot(cv_gradX_2_bw, cv_gradY_2_bw)

    cv_gradX_m2_bw = gradientX(msk2_bw)
    cv_gradY_m2_bw = gradientY(msk2_bw)
    # Get square root of sum of squares
    cv_sobel_m2_bw = np.hypot(cv_gradX_m2_bw, cv_gradY_m2_bw)

    # Gradient of Blur @ 5 - RGB
    gradX_2_5 = gradientX(blurred_img2)
    gradY_2_5 = gradientY(blurred_img2)
    # Get square root of sum of squares
    sobel_2_5 = np.hypot(gradX_2_5, gradY_2_5)

    # Gradient of Blur @ 11 - RGB
    gradX_2_11 = gradientX(blurred_img2_11)
    gradY_2_11 = gradientY(blurred_img2_11)
    # Get square root of sum of squares
    sobel_2_11 = np.hypot(gradX_2_11, gradY_2_11)

    # Gradient of Blur @ 75 - RGB
    gradX_2_75 = gradientX(blurred_img2_75)
    gradY_2_75 = gradientY(blurred_img2_75)
    # Get square root of sum of squares
    sobel_2_75 = np.hypot(gradX_2_75, gradY_2_75)

    ## BW BLURRED GRADIENTS
    # Gradient of Blur @ 5 - BW
    gradX_2_5_bw = gradientX(blurred_img2_bw)
    gradY_2_5_bw = gradientY(blurred_img2_bw)
    # Get square root of sum of squares
    sobel_2_5_bw = np.hypot(gradX_2_5_bw, gradY_2_5_bw)

    # Gradient of Blur @ 11 - BW
    gradX_2_11_bw = gradientX(blurred_img2_11_bw)
    gradY_2_11_bw = gradientY(blurred_img2_11_bw)
    # Get square root of sum of squares
    sobel_2_11_bw = np.hypot(gradX_2_11_bw, gradY_2_11_bw)

    # Gradient of Blur @ 75 - BW
    gradX_2_75_bw = gradientX(blurred_img2_75_bw)
    gradY_2_75_bw = gradientY(blurred_img2_75_bw)
    # Get square root of sum of squares
    sobel_2_75_bw = np.hypot(gradX_2_75_bw, gradY_2_75_bw)

    '''
    # AS I'm not longer using the numpy ones, it's not necessary
    diff0 = cv_sobel_0_bw-sobel_0_bw
    diff1 = cv_sobel_1_bw-sobel_1_bw
    diff2 = cv_sobel_2_bw-sobel_2_bw
    '''

    # Apply a Gradient X and Y to the SOBEL GRADIENT OF THE IMAGE
    gradX_sobel = gradientX(cv_sobel_2_bw)
    print("Minimum value of SUM is: ", np.amin(gradX_sobel))
    print("Max value of SUM is: ", np.amax(gradX_sobel))

    gradY_sobel = gradientY(cv_sobel_2_bw)
    print("-----")
    print("Minimum value of MIN is: ", np.amin(gradY_sobel))
    print("Max value of MIN is: ", np.amax(gradY_sobel))
    sum_of_grad = gradX_sobel+gradY_sobel
    min_of_grad = gradX_sobel-gradY_sobel

    '''
    print ("Minimum value of SUM is: ", np.amin(sum_of_grad))
    print("Max value of SUM is: ", np.amax(sum_of_grad))
    print("-----")
    print("Minimum value of MIN is: ", np.amin(min_of_grad))
    print("Max value of MIN is: ", np.amax(min_of_grad))

    # Before continuing, all values greater than 255, can be 255.
    # Same for values < 0

    sum_of_grad[sum_of_grad > 255] = 255
    sum_of_grad[sum_of_grad < 0] = 0

    min_of_grad[min_of_grad > 255] = 255
    min_of_grad[min_of_grad < 0] = 0

    print("--- SECOND ----")
    print("Minimum value of SUM is: ", np.amin(sum_of_grad))
    print("Max value of SUM is: ", np.amax(sum_of_grad))
    print("-----")
    print("Minimum value of MIN is: ", np.amin(min_of_grad))
    print("Max value of MIN is: ", np.amax(min_of_grad))

    '''

    # as the ndArray have values > 0
    # NORMALIZE
    var1 = np.clip(sum_of_grad, 0, 255)
    data_u8= var1.astype('uint8') #220 different values a
    # NOTE = Unique with short values = 220
    # Without shortening values = 220
    unique = np.unique(data_u8)

    # To show the histogram
    #plt.hist(data_u8.ravel(), 256, [0, 256]);
    #plt.show()

    ## Apply Mean
    mean_sum = np.mean(sum_of_grad) ## -256
    mean_min = np.mean(min_of_grad) ## 686
    mean_du8 = np.mean(data_u8)     ## 123

    #as the mean in u8 format == 123, apply this threshold
    data_u8[data_u8 <= mean_du8] = 0
    print("Minimum value of data_u8 is: ", np.amin(data_u8))
    print("Max value of SUM data_u8: ", np.amax(data_u8))
    unique = np.unique(data_u8) ## 112

    medianblur_7 = cv2.medianBlur(data_u8, ksize=7)
    medianblur_15 = cv2.medianBlur(data_u8, ksize=15)
    # ME PASSÉ: medianblur_33 = cv2.medianBlur(data_u8, ksize=33)

    ## Copied from documentation
    src = img2

    ## Canny recommended a upper:lower ratio between 2:1 and 3:1.
    dst = cv2.Canny(src, 200, 600, None, 3)

    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    ## HOUGH LINES!! >> Better to used the probabilistics
    '''
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    '''
    # dst = source
    # 1
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    mathematicalLines = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            # start point
            # distance x
            # distance y
            # degrees
            # length
            l = linesP[i][0]
            start_point = [l[0],l[1],l[2],l[3]]
            distanceX = abs(l[2]-l[0])
            distanceY = abs(l[3]-l[1])
            radians = np.degrees(math.atan2(distanceY, distanceX))
            hypotenuse = np.hypot(distanceY, distanceX)
            mathematicalLines.append([start_point, distanceX, distanceY, radians, hypotenuse])
            #cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    verticalLines = []
    horitzontalLines = []
    otherLines = []
    for i in range(0,len(mathematicalLines)):
        if (-0.5 <= mathematicalLines[i][3] <= 0.5):
            horitzontalLines.append(mathematicalLines[i])
        elif (89.5 <= mathematicalLines[i][3] <= 90.5):
            verticalLines.append(mathematicalLines[i])
        else:
            otherLines.append(mathematicalLines[i])

    for i in range(0, len(horitzontalLines)):
        #print horitzontal in blue
        #image_to_be_drawn  // Start point // End Point // color // thickness //
        cv2.line(cdstP, (horitzontalLines[i][0][0], horitzontalLines[i][0][1]), (horitzontalLines[i][0][2], horitzontalLines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

    for i in range(0, len(verticalLines)):
        #print horitzontal in blue
        #image_to_be_drawn  // Start point // End Point // color // thickness //
        cv2.line(cdstP, (verticalLines[i][0][0], verticalLines[i][0][1]),
                 (verticalLines[i][0][2], verticalLines[i][0][3]), (0, 255, 0), 3, cv2.LINE_AA)

    for i in range(0, len(otherLines)):
        # print horitzontal in blue
        # image_to_be_drawn  // Start point // End Point // color // thickness //
        cv2.line(cdstP, (otherLines[i][0][0], otherLines[i][0][1]),
                 (otherLines[i][0][2], otherLines[i][0][3]), (255, 0, 0), 3, cv2.LINE_AA)



    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv2.waitKey()

    print("hey, stop!")
