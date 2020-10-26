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


    ## These gradients are much better. More general (not tinny detailed)
    ## GRADIENTS IN BW  [USING CV]
    cv_gradX_0_bw = gradientX(img0_bw)
    cv_gradY_0_bw = gradientY(img0_bw)
    # Get square root of sum of squares
    cv_sobel_0_bw = np.hypot(cv_gradX_0_bw, cv_gradY_0_bw)

    cv_gradX_1_bw = gradientX(img1_bw)
    cv_gradY_1_bw = gradientY(img1_bw)
    cv_sobel_1_bw = np.hypot(cv_gradX_1_bw, cv_gradY_1_bw)

    cv_gradX_2_bw = gradientX(img2_bw)
    cv_gradY_2_bw = gradientY(img2_bw)
    cv_sobel_2_bw = np.hypot(cv_gradX_2_bw, cv_gradY_2_bw)

    cv_gradX_m2_bw = gradientX(msk2_bw)
    cv_gradY_m2_bw = gradientY(msk2_bw)
    cv_sobel_m2_bw = np.hypot(cv_gradX_m2_bw, cv_gradY_m2_bw)

    # Gradient of Blur @ 5 - RGB
    gradX_2_5 = gradientX(blurred_img2)
    gradY_2_5 = gradientY(blurred_img2)
    sobel_2_5 = np.hypot(gradX_2_5, gradY_2_5)

    # Gradient of Blur @ 11 - RGB
    gradX_2_11 = gradientX(blurred_img2_11)
    gradY_2_11 = gradientY(blurred_img2_11)
    sobel_2_11 = np.hypot(gradX_2_11, gradY_2_11)

    # Gradient of Blur @ 75 - RGB
    gradX_2_75 = gradientX(blurred_img2_75)
    gradY_2_75 = gradientY(blurred_img2_75)
    sobel_2_75 = np.hypot(gradX_2_75, gradY_2_75)

    ## BW BLURRED GRADIENTS
    # Gradient of Blur @ 5 - BW
    gradX_2_5_bw = gradientX(blurred_img2_bw)
    gradY_2_5_bw = gradientY(blurred_img2_bw)
    sobel_2_5_bw = np.hypot(gradX_2_5_bw, gradY_2_5_bw)

    # Gradient of Blur @ 11 - BW
    gradX_2_11_bw = gradientX(blurred_img2_11_bw)
    gradY_2_11_bw = gradientY(blurred_img2_11_bw)
    sobel_2_11_bw = np.hypot(gradX_2_11_bw, gradY_2_11_bw)

    # Gradient of Blur @ 75 - BW
    gradX_2_75_bw = gradientX(blurred_img2_75_bw)
    gradY_2_75_bw = gradientY(blurred_img2_75_bw)
    sobel_2_75_bw = np.hypot(gradX_2_75_bw, gradY_2_75_bw)

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


    # as the ndArray have values > 0
    # NORMALIZE
    var1 = np.clip(sum_of_grad, 0, 255)
    data_u8= var1.astype('uint8') #220 different values a
    # NOTE = Unique with short values = 220
    # Without shortening values = 220
    unique = np.unique(data_u8)

    # To show the histogram
    ## If you want, you can use it.... pero el codi estava per a veure histogrames i comprovar
    ## que les modificacions anteriors estaven bé
    #plt.hist(data_u8.ravel(), 256, [0, 256]);
    #plt.show()

    ## Apply Mean
    ## Els valors al costat son indicatius del valor inicial
    mean_sum = np.mean(sum_of_grad) ## -256
    mean_min = np.mean(min_of_grad) ## 686
    mean_du8 = np.mean(data_u8)     ## 123

    #as the mean in u8 format == 123, apply this threshold
    ## En la seguent linia, per a una imatge data_u8 que té integer_uint8
    ## d'aqui el seu nom... tots els valors inferiors al seu threshold,
    ## son posats a 0.
    ## aquest és un exemple de modificació / binarització de tota la imatge
    ## fet de forma manual.
    data_u8[data_u8 <= mean_du8] = 0
    #print("Minimum value of data_u8 is: ", np.amin(data_u8))
    #print("Max value of SUM data_u8: ", np.amax(data_u8))
    unique = np.unique(data_u8) ## 112 -- Quins son els valors que son unics en una imatge que te la meitat = 0

    medianblur_7 = cv2.medianBlur(data_u8, ksize=7)
    medianblur_15 = cv2.medianBlur(data_u8, ksize=15)
    # ME PASSÉ: medianblur_33 = cv2.medianBlur(data_u8, ksize=33)

    ## CODI FUNCIÓ HOUGH LINES!!
    ## A PARTIR D'AQUIÍ ÉS on he obtingut millors resultats


    ## Copied from documentation: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    ## Torno a carregar la imatge.
    ## Tot aquest codi prové de la documentació: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    ## Si, se que ho he posat 2 vegades. Es perquè quedi clar!
    src = img2

    ## Canny recommended a upper:lower ratio between 2:1 and 3:1. (from documentation)
    ## TODO: Pots jugar amb els valors de Canny, ja que no els he tocat i m'han funcionat per a la imatge inicial
    ## Però hi jugaràs, si en altres imatges veus que Canny va malament i no et pilla cap linia del quadre
    dst = cv2.Canny(src, 200, 600, None, 3)

    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    ## HOUGH LINES:
    ## Hi has dos versions. La 2a es la priemra que he trobat. La seguent es la que estic fent servir
    ## Fora bo anar comparant una o altre.... i jugar amb els caracters
    linesP = cv2.HoughLinesP(dst, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=100, maxLineGap=10)
    #linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    # SOURCE size = 140 lines  -- ANotacio utilitzada per veure si anava millorant la detecció de linies
    # mathematicalLines es un array que intenta expressar les linies d'una manera més humana
    # amb punts d'inici, final, graus, distàncies....
    mathematicalLines = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            start_point = [l[0],l[1],l[2],l[3]]
            distanceX = abs(l[2]-l[0])
            distanceY = abs(l[3]-l[1])
            radians = np.degrees(math.atan2(distanceY, distanceX))
            hypotenuse = np.hypot(distanceY, distanceX)
            mathematicalLines.append([start_point, distanceX, distanceY, radians, hypotenuse])
            #cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    verticalLines = [] # SOURCE = 5 || 4
    horitzontalLines = [] # SOURCE = 17  || 10
    otherLines = [] # SOURCE = 13 || 3
    for i in range(0,len(mathematicalLines)):
        if (-2 <= mathematicalLines[i][3] <= 2):
            horitzontalLines.append(mathematicalLines[i])
        elif (88 <= mathematicalLines[i][3] <= 92):
            verticalLines.append(mathematicalLines[i])
        else:
            otherLines.append(mathematicalLines[i])
    print ("TOTAL of lines: ", len(mathematicalLines))

    ## Lines are defined by:
    ## tuple of 4 point  [x1, y1, x2, y2]
    ## >> Note, when creating, you need to create two points : [ P1 (x1,y1) ; P2 (x2, y2) ]
    ## distance in X [for vertical Lines, should be close to 0]
    ## distance in y [for horitzontal lines, should be close to 0]
    ## degrees [of course you can use grad..... but degrees are great]
    ## lenght [

    for i in range(0, len(horitzontalLines)):
        #print horitzontal in RED
        #image_to_be_drawn  // Start point // End Point // color // thickness // AA = Maco (no ho toquis) (opcions: 4 pixels, 8 pixels)
        cv2.line(cdstP, (horitzontalLines[i][0][0], horitzontalLines[i][0][1]), (horitzontalLines[i][0][2], horitzontalLines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

    print ("Horitzontal lines: ", len(horitzontalLines))

    for i in range(0, len(verticalLines)):
        #print VERTICAL in GREEN
        cv2.line(cdstP, (verticalLines[i][0][0], verticalLines[i][0][1]),
                 (verticalLines[i][0][2], verticalLines[i][0][3]), (0, 255, 0), 3, cv2.LINE_AA)
        print("LINE ", i, ": ")
        print("--------------")
        print(verticalLines[i])

    print("Vertical lines: ", len(verticalLines))

    for i in range(0, len(otherLines)):
        # print OTHERS in BLUE
        cv2.line(cdstP, (otherLines[i][0][0], otherLines[i][0][1]),
                 (otherLines[i][0][2], otherLines[i][0][3]), (255, 0, 0), 3, cv2.LINE_AA)

    print("Other lines: ", len(otherLines))


    ## Ho comento perquè no és necessari
    ## cv2.imshow("Source", src) # Imatge original
    cv2.imshow("Detected Lines - Probabilistic Line Transform", cdstP)

    cv2.waitKey()

    print("hey, stop!")
