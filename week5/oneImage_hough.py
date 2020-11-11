import argparse
from os import path
from time import time
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2

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

def edge_segmentation(img):
    """ Detect edges to create a mask that indicates where the paintings are located """
    sx, sy = np.shape(img)[:2]
    datatype = np.uint8

    kernel = np.ones((15,15), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(img, 10, 80, None, 3)
    
    # Closing to ensure edges are continuous
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Filling
    kernel = np.ones((15,15), dtype=np.uint8)
    mask = (ndimage.binary_fill_holes(edges)).astype(np.float64)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1,int(mask.shape[1]*0.05))))

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[:, -1]

    top_two_conn_comp_idx = sizes.argsort()
    top_two_conn_comp_idx = top_two_conn_comp_idx[top_two_conn_comp_idx!=0]
    idxs_tt = ((np.arange(0, min(3, len(top_two_conn_comp_idx)))+1)*(-1))[::-1]
    top_two_conn_comp_idx = top_two_conn_comp_idx[idxs_tt][::-1]
    
    idxs = [idx for idx in top_two_conn_comp_idx]

    bc = np.zeros(output.shape)
    bc[output == idxs[0]] = 255
    bc = create_convex_painting(mask, bc)
    #cv2.waitKey(0)

    #bc = refine_mask(img, bc, get_bbox(bc))

    if len(idxs) > 1:
        sbc = np.zeros(output.shape)
        sbc[output == idxs[1]] = 255
        sbc = create_convex_painting(mask, sbc)
        #if sbc.astype(np.uint8).sum() > 0:
        #    sbc = refine_mask(img, sbc, get_bbox(sbc))

        if len(idxs) > 2:
            tbc = np.zeros(output.shape)
            tbc[output == idxs[2]] = 255
            tbc = create_convex_painting(mask, tbc)
            #if tbc.astype(np.uint8).sum() > 0:
            #    tbc = refine_mask(img, tbc, get_bbox(tbc))

    resulting_masks = bc
        
    return resulting_masks

def create_convex_painting(mask, component_mask):
    kernel = np.ones((5, 5), np.uint8)
    
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, borderValue=0)
    contours, hierarchy = cv2.findContours((component_mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask).astype(np.uint8)
    polished_mask = cv2.fillPoly(mask, contours, 255).astype(np.uint8)
    return polished_mask

# Reusing some calls from the MAIN code,
# this program focused in one imag
if __name__ == "__main__":
    # Path to DB image:
    my_path = "/home/sergio/MCV/M1/Practicas/DB"
    path0 = "qsd1_w5/00000.jpg"
    path1 = "qsd1_w5/00000.jpg"

    """ 
    path0 = absolutePath(path0)
    print ("Image from the DB: ", path0)
    # Path to image in QDS1
    
    path1 = absolutePath(path1)
    # Path to image in QDS2
    path2 = "qsd1_w5/00008.jpg"
    path2 = absolutePath(path2)
    # Path to mask in QSD2
    path3 = "qsd2_w2/00000.png"
    path3 = absolutePath(path3)
    """
    path0 = path.join(my_path, path0)
    path1 = path.join(my_path, path1)
    img0 = openImage(path0)
    img1 = openImage(path1)
    #img2 = openImage(path2)

    #msk2 = openImage(path3)

    ## Copied from documentation: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    ## Torno a carregar la imatge.
    ## Tot aquest codi prové de la documentació: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    ## Si, se que ho he posat 2 vegades. Es perquè quedi clar!
    img0 = cv2.medianBlur(img0, 5)
    img1 = cv2.medianBlur(img1, 5)

    img1 = edge_segmentation(img1)
    img1 = cv2.morphologyEx(img1,cv2.MORPH_GRADIENT, np.ones((3,3)))

    src = cv2.resize(img1, (512,512))

    # Method 1 -> HoughLinesP
    # Method 2 -> HoughLines
    method = 2

    ## Canny recommended a upper:lower ratio between 2:1 and 3:1. (from documentation)
    ## TODO: Pots jugar amb els valors de Canny, ja que no els he tocat i m'han funcionat per a la imatge inicial
    ## Però hi jugaràs, si en altres imatges veus que Canny va malament i no et pilla cap linia del quadre
    dst = cv2.Canny(src, 20, 100, None, 3)
    #kernel = np.ones((15,15), dtype=np.uint8)
    #edges = cv2.dilate(dst, kernel, iterations=1)
    #edges = cv2.erode(edges, kernel, iterations=1)

    cdst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    #cv2.imshow("Detected Lines - Hough Line Transform", cv2.resize(cdstP,(512,512)))
    #cv2.waitKey(0)
    
    
    ## HOUGH LINES:
    ## Hi has dos versions. La 2a es la priemra que he trobat. La seguent es la que estic fent servir
    ## Fora bo anar comparant una o altre.... i jugar amb els caracters
    if method == 1:
        linesP = cv2.HoughLinesP(dst, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=int(dst.shape[0]*0.01), maxLineGap=int(dst.shape[0]*0.1))
        #linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        # SOURCE size = 140 lines  -- ANotacio utilitzada per veure si anava millorant la detecció de linies
        # mathematicalLines es un array que intenta expressar les linies d'una manera més humana
        # amb punts d'inici, final, graus, distàncies....
        print(linesP)
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
            if (-40 <= mathematicalLines[i][3] <= 40):
                horitzontalLines.append(mathematicalLines[i])
            elif (60 <= mathematicalLines[i][3] <= 150):
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

        for i in range(0, len(otherLines)):
            # print OTHERS in BLUE
            cv2.line(cdstP, (otherLines[i][0][0], otherLines[i][0][1]),
                    (otherLines[i][0][2], otherLines[i][0][3]), (255, 0, 0), 3, cv2.LINE_AA)

        print("Other lines: ", len(otherLines))

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

    elif method == 2:
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 75, None, 0, 0)
        #linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        # SOURCE size = 140 lines  -- ANotacio utilitzada per veure si anava millorant la detecció de linies
        # mathematicalLines es un array que intenta expressar les linies d'una manera més humana
        # amb punts d'inici, final, graus, distàncies....
        valid_lines = []
        
        for line in lines:
            rho,theta = line[0]
            if theta > math.pi:
                theta = theta-math.pi
            if theta > math.pi/2 + math.pi/4:
                theta = theta-math.pi

            assigned = False
            for i in range(len(valid_lines)):
                mean_rho, mean_theta = np.array(valid_lines[i]).mean(axis=0)
                
                if abs(mean_theta-theta) < math.pi/4 and abs(mean_rho-rho) < (dst.shape[0])*0.2:
                    valid_lines[i].append([rho, theta])
                    assigned = True
                    break
            
            if not assigned:
                valid_lines.append([[rho, theta]])

        lines = [np.array(v).mean(axis=0).tolist() for v in valid_lines]
        lines_to_proc = []
        for line in lines:
            if line[1] < 0:
                line[1] = line[1]+math.pi
            lines_to_proc.append(line)

        lines = lines_to_proc

        result_intersect = np.zeros((cdst.shape[0], cdst.shape[1]))

        for line in lines:
            rho,theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
            x1 = int(x0 + 1000 * (-b))
            # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
            y1 = int(y0 + 1000 * (a))
            # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
            x2 = int(x0 - 1000 * (-b))
            # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
            y2 = int(y0 - 1000 * (a))
            tmp = np.zeros((cdstP.shape[0], cdstP.shape[1]))
            
            tmp = cv2.line(tmp, (x1, y1), (x2, y2), 255, 1)
            cdstP = cv2.line(cdstP, (x1, y1), (x2, y2), (255,0,0), 1)
            result_intersect[tmp!=0] += 1

        positions = np.where(result_intersect > 1)

        # Find ...
        positions = np.array(positions)

        ss = positions[0,:]**2+positions[1,:]**2
        first_pos = np.where(ss == ss.min())[0][0]
        topl_x, topl_y = positions[:, first_pos]

        ss = (cdstP.shape[0]-positions[0,:])**2+positions[1,:]**2
        first_pos = np.where(ss == ss.min())[0][0]
        topr_x, topr_y = positions[:, first_pos]

        ss = (cdstP.shape[0]-positions[0,:])**2+(cdstP.shape[1]-positions[1,:])**2
        first_pos = np.where(ss == ss.min())[0][0]
        botr_x, botr_y = positions[:, first_pos]

        ss = positions[0,:]+(cdstP.shape[1]-positions[1,:])**2
        first_pos = np.where(ss == ss.min())[0][0]
        botl_x, botl_y = positions[:, first_pos]

        y = botl_y-botr_y
        x = botr_x-botl_x
        angle = (math.atan(np.clip(y/x, 0, 1))*int(y>0))*(180/math.pi)

        print(positions, angle)

    ## Ho comento perquè no és necessari
    ## cv2.imshow("Source", src) # Imatge original
    cv2.imshow("Detected Lines - Hough Line Transform", cdstP)

    cv2.waitKey(0)

    print("hey, stop!")
