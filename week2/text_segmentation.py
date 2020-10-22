import numpy as np
import cv2
from background_mask import get_bbox
from debug_utils import *

def estimate_text_mask(cropped_imgs, method):
    """ List of list of images. Each list contains one element for each detected painting in the image.
        params:
            imgs: [[painting1, painting2], [painting1], [painting1, painting2], ...]
            method: text segmentation method 
        return: images text boolean mask
    """
    text_segm = get_method(method)
    cropped_text_mask = []
    bboxes_mask = []
    for paintings in cropped_imgs:
        for painting in paintings:
            result = text_segm(painting).astype(bool)
            cropped_imgs.append(result[0])
            bboxes_mask.append(result[1])

    return cropped_text_mask, bboxes_mask

def crop_painting_for_text(imgs, bboxes):
    """ Rectangular paintings for images """
    print(bboxes[0])
    rectangular_crops = []
    img_num = 0
    for bboxes_painting in bboxes:
        painting_boxes = []
        # TODO: Fix in QSD1
        # BBOX = 0 == INT >> NOT SUSCRIPTABLE
        # Error raised:
        # TypeError: 'int' object is not subscriptable
        try:
            for bbox in bboxes_painting:
                if isDebug():
                    print("Range X: ", bbox[0], ":", bbox[2])
                    print("Range Y: ", bbox[1], ":", bbox[3])
                print("Painting_boxes before: ", painting_boxes)
                painting_boxes.append(imgs[img_num][bbox[0]:bbox[2], bbox[1]:bbox[3]])
                print("Painting_boxes after: ", painting_boxes)
        except TypeError:
            print("OOOPS, This has gone too far");
        img_num += 1
        rectangular_crops.append(painting_boxes)
    return rectangular_crops

def f_pixel_cone(pix): 
    """ Pixel-wise triangular transformation """
    if (pix < 127): 
        return pix*2 
    else:
        return abs(255-pix)

def get_channel(img, ch=2):
    """ """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img = img[:,:,ch]
    vectorized_pixelwise_op = np.vectorize(f_pixel_cone)
    img = vectorized_pixelwise_op(img)
    return img

# Define your methods here
def saturation_masking(img):
    trl2 = 0.01
    tiny_region_letters = 0.015
    big_region_box = 0.05

    """ img_s = get_channel(img.copy(), 2)
    img_c = img_s.copy().reshape(-1)
    img_c.sort()
    s_threshold = img_c[int(0.1*len(img_c))]
    mask = img_s < s_threshold

    
    s1,s2 = int(img.shape[0]*0.01) , int(img.shape[1]*0.01)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, borderValue=0)

    s1,s2 = int(img.shape[0]*tiny_region_letters) , int(img.shape[1]*tiny_region_letters)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_CLOSE, kernel, borderValue=0)

    s1,s2 = int(img.shape[0]*big_region_box), int(img.shape[1]*big_region_box)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_OPEN, kernel, borderValue=0) """

    # TODO: Select most adequate box
    """ img_s = get_channel(img.copy(), 1)
    img_c = img_s.copy().reshape(-1)
    img_c.sort()
    s_threshold = img_c[int(0.1*len(img_c))]
    polished_mask2 = img_s < s_threshold

    s1,s2 = int(img.shape[0]*0.01) , int(img.shape[1]*0.01)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask2 = cv2.morphologyEx(polished_mask2.astype(np.uint8), cv2.MORPH_OPEN, kernel, borderValue=0)

    s1,s2 = int(img.shape[0]*tiny_region_letters) , int(img.shape[1]*tiny_region_letters)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask2 = cv2.morphologyEx(polished_mask2*255, cv2.MORPH_CLOSE, kernel, borderValue=0)

    s1,s2 = int(img.shape[0]*big_region_box), int(img.shape[1]*big_region_box)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask2 = cv2.morphologyEx(polished_mask2*255, cv2.MORPH_OPEN, kernel, borderValue=0) """
    polished_mask = get_mask_ch(img, 1)
    polished_mask2 = get_mask_ch(img, 2)


    cv2.imshow("img", cv2.resize(img,(512,512)))
    cv2.imshow("text_mask", cv2.resize(polished_mask,(512,512)))
    cv2.imshow("text_mask2", cv2.resize(polished_mask2,(512,512)))
    mask_best1, max_score1 = get_best_bbox(polished_mask)
    mask_best2, max_score2 = get_best_bbox(polished_mask2)
    cv2.imshow("text_mask_max", cv2.resize(mask_best1.astype(np.uint8)*255,(512,512)))
    cv2.imshow("text_mask_max2", cv2.resize(mask_best2.astype(np.uint8)*255,(512,512)))
    print(max_score1, max_score2)

    cv2.waitKey(0)

    return polished_mask

def get_mask_ch(img, ch):
    img_s = get_channel(img, ch)
    img_c = img_s.copy().reshape(-1)
    img_c.sort()
    s_threshold = img_c[int(0.2*len(img_c))]
    mask = img_s < s_threshold

    tiny_region_letters = 0.015
    big_region_box = 0.05

    s1, s2 = int(img.shape[0]*0.005), int(img.shape[0]*0.01) #int(img.shape[0]*0.02), int(img.shape[1]*0.02)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, borderValue=0)

    s1, s2 = int(img.shape[0]*tiny_region_letters), int(img.shape[1]*tiny_region_letters)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_CLOSE, kernel, borderValue=0)

    s1, s2 = int(img.shape[0]*0.01), int(img.shape[0]*0.02) #int(img.shape[0]*big_region_box), int(img.shape[1]*big_region_box)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_OPEN, kernel, borderValue=0)

    s1, s2 = 7, 7 #int(img.shape[0]*big_region_box), int(img.shape[1]*big_region_box)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_OPEN, kernel, borderValue=0)
    
    positions = np.where(polished_mask==255)
    min_w, max_w = positions[1].min(), positions[1].max()

    s1, s2 = 5, int((max_w-min_w)/2) #int(img.shape[0]*big_region_box), int(img.shape[1]*big_region_box)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_OPEN, kernel, borderValue=0)
    
    
    return polished_mask

def get_best_bbox(mask):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[:, -1]

    top_cc_idx = sizes.argsort()
    idxs_reversed = np.arange(1, min(20, len(top_cc_idx)))*(-1)
    top_cc_idx = top_cc_idx[top_cc_idx!=0][idxs_reversed][::-1]

    max_idx = -1
    max_score = -1
    
    for idx in top_cc_idx:
        mask_box = (output == idx).astype(np.uint8)*255
        x0,y0,x1,y1 = get_bbox(mask_box)
        
        sum_pixels = (mask_box == 255).astype(np.uint8).sum()
        score = sum_pixels/((x1-x0)*(y1-y0))*(sum_pixels/(mask.shape[0]*mask.shape[1]))
        if score > max_score:
            max_score = score
            max_idx = idx

    return output == max_idx, max_score



def test(image):
    
    saturation_threshold = 5

    # Color image segmentation to create binary image (255 white: high possibility of text; 0 black: no text)
    im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(im_hsv)

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_grey[s < saturation_threshold] = 255
    image_grey[image_grey != 255] = 0

    # Cleaning image using morphological opening filter
    opening_kernel = np.ones((15, 10),np.uint8)
    text_mask = cv2.morphologyEx(image_grey, cv2.MORPH_OPEN, opening_kernel, iterations=1)


    #------------------------------   FINDING AND CHOOSING CONTOURS OF THE BINARY MASK   ---------------------------------------

    # Finding contours of the white areas of the images (high possibility of text)
    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Initialize parameters
    largest_area, second_largest_area, x_box_1, y_box_1, w_box_1, h_box_1, x_box_2, y_box_2, w_box_2, h_box_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    image_width = text_mask.shape[0]

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if ((w/h > 2) & (w/h < 12) & (w > (0.1 * image_width)) & (area > second_largest_area)):

            if area > largest_area:
                x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
                x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                second_largest_area = largest_area
                largest_area = area

            else:
                x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                second_largest_area = area
    
    result_mask = np.zeros((image.shape[0], image.shape[1]))
    result_mask[y_box_1:y_box_1+h_box_1, x_box_1:x_box_1+w_box_1] = 255
    cv2.imshow("img", image)
    cv2.imshow("res", result_mask)
    cv2.waitKey(0)
    return [2,3]

# Selection utils
SATURATION_MASKING = "SM"
MM = "MM"
OPTIONS = [SATURATION_MASKING, MM]

METHOD_MAPPING = {
    OPTIONS[0]: saturation_masking,
    OPTIONS[1]: test
}

def get_method(method):
    return METHOD_MAPPING[method]
