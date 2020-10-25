import numpy as np
import cv2
from background_mask import get_bbox
from debug_utils import *
from tqdm import tqdm
from collections import defaultdict

def estimate_text_mask(cropped_imgs, painting_bboxes, method, qs_images):
    """ List of list of images. Each list contains one element for each detected painting in the image.
        params:
            imgs: [[painting1, painting2], [painting1], [painting1, painting2], ...]
            method: text segmentation method 
        return: images text boolean mask
    """
    text_segm = get_method(method)
    cropped_text_mask = []
    bboxes_mask = []
    bbox_show = []

    print("Extracting text boxes...")
    for paintings, pantings_bboxes in tqdm(zip(cropped_imgs,painting_bboxes)):
        bboxes_paintings = []
        cropped_text_mask_paintings = []
        bbox_show_paintings = []

        for painting, painting_bbox in zip(paintings, pantings_bboxes):
            result = text_segm(painting)
            cropped_text_mask_paintings.append(result[0]) #.astype(bool))
            bbox_show_paintings.append(result[1])

            bbox_relative = result[1]
            bbox_relative = [bbox_relative[0]+painting_bbox[1], bbox_relative[1]+painting_bbox[0], \
                             bbox_relative[2]+painting_bbox[1], bbox_relative[3]+painting_bbox[0]]
            bboxes_paintings.append(bbox_relative)

        cropped_text_mask.append(cropped_text_mask_paintings)
        bboxes_mask.append(bboxes_paintings)
        bbox_show.append(bbox_show_paintings)

    if isDebug():
        # Show images
        i = 0
        for paintings in cropped_imgs:
            j = 0
            for painting in paintings:
                bbox = bbox_show[i][j]
                painting_copy = painting.copy()
                painting_copy = cv2.rectangle(painting_copy, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0 ,0), 10)
                cv2.imshow("result text segm", cv2.resize(painting_copy,(512,512)))
                cv2.waitKey(0)
                
                j += 1
            i += 1

    return [cropped_text_mask, bboxes_mask]

def crop_painting_for_text(imgs, bboxes):
    """ Rectangular paintings for images """
    rectangular_crops = []
    img_num = 0

    for bboxes_painting in bboxes:
        painting_boxes = []
        for bbox in bboxes_painting:
            painting_boxes.append(imgs[img_num][bbox[0]:bbox[2], bbox[1]:bbox[3]])

        img_num += 1
        rectangular_crops.append(painting_boxes)
    return rectangular_crops

def f_pixel_cone(pix): 
    """ Pixel-wise triangular transformation """
    if (pix < 127): 
        return pix*2 
    else:
        return abs(255-pix)

def compute_score_mask(mask_box):
    try:
        x0,y0,x1,y1 = get_bbox(mask_box)
    except:
        return -1
    
    sum_pixels = (mask_box == 255).astype(np.uint8).sum()
    score = sum_pixels/((x1-x0)*(y1-y0))*(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))
    return score

def resize_img(img):
    return cv2.resize(img, (512,512))

def imshow(name, img):
    cv2.imshow(name, resize_img(img))

def morphological_method1(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s1,s2 = img.shape[0]//20, img.shape[1]//20

    # Find text/box region
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)), borderType=cv2.BORDER_REPLICATE)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)), borderType=cv2.BORDER_REPLICATE)
    tophat = closing - opening

    m1, m2 = int(0.05*img.shape[0]), int(0.05*img.shape[1])
    tophat = tophat[m1:img.shape[0]-m1, m2:img.shape[1]-m2]

    # Binarize based on maximum element
    thr = 0.98
    copy_th = tophat.copy().reshape(-1)
    copy_th.sort()
    thresh = tophat > copy_th[int(thr*len(copy_th))]

    # Clean noise
    thresh = cv2.morphologyEx(thresh.astype(np.uint8)*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,3)))
    s_v = int(img.shape[1]*0.02)
    
    # Add padding
    padding = 250 
    dilation = cv2.copyMakeBorder(src=thresh, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0) 

    # Close with a wide structuring element
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[1]*0.6), 11)), borderValue=0)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[1]*0.6), 11)), borderValue=0)
    
    # Remove padding
    text_mask = dilation[padding:dilation.shape[0]-padding, padding:dilation.shape[1]-padding]
    
    # Finding contours of the white areas of the images (high possibility of text)
    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # Initialize parameters
    xb1, yb1, wb1, hb1 = 0, 0, 0, 0
    max_score = -1

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        x0,y0,x1,y1 = x, y, x+w, y+h 
        score = compute_score_mask_text(text_mask, [x0,y0,x1,y1])

        if score > max_score and w > 0.1*text_mask.shape[1] and (w/h > 1.5) and (w/h < 40):
            xb1, yb1, wb1, hb1 = x, y, w, h
            max_score = score

    mask = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    w_e, h_e = int((wb1)*0.01), int((hb1)*0.4) # Extra height to account for errors

    pos = [m1+xb1-w_e, m2+yb1-h_e, xb1 + m1 + wb1 + w_e, m2 + yb1 + hb1 + h_e]
    mask[max(pos[1],0):min(pos[3], mask.shape[0]), max(pos[0],0):min(pos[2], mask.shape[1])] = 255
    return mask, pos

def compute_score_mask_text(mask_box, bbox):
    x0,y0,x1,y1 = bbox
    mask_box_part = mask_box[y0:y1,x0:x1]
    sum_pixels = (mask_box_part == 255).astype(np.uint8).sum()
    md = max(mask_box.shape[0], mask_box.shape[1])
    score = (1-abs(0.3-(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))))+\
            ((mask_box.shape[0]*mask_box.shape[1])/(md*md))+0.2*sum_pixels/((y1-y0)*(x1-x0))
    return score

def compute_score_mask2(mask_box, bbox):
    x0,y0,x1,y1 = bbox
    sum_pixels = (mask_box[y0:y1, x0:x1] == 255).astype(np.uint8).sum()
    
    mask_box_cp = mask_box.copy()
    mean_x = int(np.where(mask_box_cp==255)[1].mean()) 
    score = 0.2*sum_pixels/((x1-x0)*(y1-y0))+\
        0.8*(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))+\
        1*abs((mask_box.shape[1]//2)-mean_x)/(mask_box.shape[1]//2)
    return score

def best_segmentation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_sat = hsv[:,:,1]

    # Clean noise
    img_sat = cv2.medianBlur(img_sat, 5)
    kernel = np.ones((1,30)) 
    img_sat = cv2.morphologyEx(img_sat, cv2.MORPH_OPEN, kernel)

    # Convert the image to binary
    morph_cp = img_sat.copy()
    morph_cp = morph_cp.reshape(-1)
    morph_cp.sort()
    th1 = img_sat < morph_cp[int(len(morph_cp)*0.075)]+5
    
    th_m1,th_m2 = int(0.025*th1.shape[0]), int(0.1*th1.shape[1])
    th1[:th_m1, :] = False
    th1[th1.shape[0]-th_m1:, :] = False
    th1[:, :th_m2] = False
    th1[:, th1.shape[1]-th_m2:] = False
    th1 = th1.astype(np.uint8)*255

    # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
    padding = max(img.shape[0], img.shape[1])
    th1 = cv2.copyMakeBorder(src=th1, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0) 
    
    kernel = np.ones((7, 1), np.uint8)
    text_mask = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, img.shape[1] // 16), np.uint8) 
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((img.shape[0] // 64, img.shape[1] // 4), np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)
    text_mask = text_mask[padding:-padding, padding:-padding]

    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Initialize parameters
    largest_score, x_box_1, y_box_1, w_box_1, h_box_1 = 0, 0, 0, 0, 0

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        x0,y0,x1,y1 = x, y, x+w, y+h 
        score = compute_score_mask2(text_mask, [x0,y0,x1,y1])
        if (w / h > 2) and (w / h < 30) and (w > (0.1 * text_mask.shape[0])) and score > largest_score:
            x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
            x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
            largest_score = score

    mask = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    pos = [x_box_1, y_box_1, x_box_1 + w_box_1 , y_box_1 + h_box_1]
    mask[max(pos[1],0):min(pos[3], mask.shape[0]), max(pos[0],0):min(pos[2], mask.shape[1])] = 255
    return mask, pos

# Selection utils
MM = "MM"
MM2 = "MM2"
OPTIONS = [MM, MM2]

METHOD_MAPPING = {
    OPTIONS[0]: morphological_method1,
    OPTIONS[1]: best_segmentation
}

def get_method(method):
    return METHOD_MAPPING[method]
