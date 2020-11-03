import numpy as np
import cv2
from background_mask import get_bbox
from debug_utils import *
from tqdm import tqdm
from collections import defaultdict
from evaluation.mask_evaluation import bb_intersection_over_union, bb_int_a_over_b

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
    relative_boxes = []
    bbox_show = []

    print("Extracting text boxes...")
    for paintings, pantings_bboxes in tqdm(zip(cropped_imgs,painting_bboxes)):
        bboxes_paintings = []
        cropped_text_mask_paintings = []
        bbox_show_paintings = []
        relative_boxes_paintings = []

        for painting, painting_bbox in zip(paintings, pantings_bboxes):
            result = text_segm(painting)
            cropped_text_mask_paintings.append(result[0]) #.astype(bool))
            bbox_show_paintings.append(result[1])

            bbox_relative = result[1]
            relative_boxes_paintings.append(bbox_relative.copy())
            bbox_relative = [bbox_relative[0]+painting_bbox[1], bbox_relative[1]+painting_bbox[0], \
                             bbox_relative[2]+painting_bbox[1], bbox_relative[3]+painting_bbox[0]]
            bboxes_paintings.append(bbox_relative)

        cropped_text_mask.append(cropped_text_mask_paintings)
        bboxes_mask.append(bboxes_paintings)
        bbox_show.append(bbox_show_paintings)
        relative_boxes.append(relative_boxes_paintings)

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

    return [cropped_text_mask, bboxes_mask, relative_boxes]

def process_gt_text_mask(qs_text_boxes, painting_bboxes, qs_imgs):
    cropped_text_mask = []
    relative_bboxes = []

    for text_boxes, painting_boxes in zip(qs_text_boxes, painting_bboxes):
        cr_text_mask = []
        rl_boxes = []
        text_boxes_to_assign = [tbox for tbox in text_boxes]
        painting_boxes_to_assign = [pbox for pbox in painting_boxes]

        for painting_box in painting_boxes: # text_boxes_to_assign
            
            best_candidate = None
            best_iou = -1
            best_idx = 0
            idx = 0
            for tbox in text_boxes_to_assign:
                c_iou = bb_int_a_over_b([tbox[1], tbox[0], tbox[3], tbox[2]], painting_box)
                if c_iou > best_iou:
                    best_iou = c_iou
                    best_candidate = tbox
                    best_idx = idx
                idx += 1

            del text_boxes_to_assign[best_idx]
            text_box = best_candidate
            mask = np.zeros((painting_box[2]-painting_box[0], painting_box[3]-painting_box[1]), dtype=np.uint8)
            mask[text_box[1]:text_box[3], text_box[0]:text_box[2]] = 255
            cr_text_mask.append(mask)
            relative_box = [max(0, text_box[0]-painting_box[1]), max(0, text_box[1]-painting_box[0]), max(1, text_box[2]-painting_box[1]), max(text_box[3]-painting_box[0], 1)]
            rl_boxes.append(relative_box)

        cropped_text_mask.append(cr_text_mask)
        relative_bboxes.append(rl_boxes)

    if isDebug():
        # Show images
        i = 0
        for paintings in qs_imgs:
            j = 0
            for painting in paintings:
                bbox = relative_bboxes[i][j]
                painting_copy = painting.copy()
                painting_copy = cv2.rectangle(painting_copy, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0 ,0), 10)
                cv2.imshow("result text segm", cv2.resize(painting_copy,(512,512)))
                cv2.waitKey(0)
                
                j += 1
            i += 1

    return [cropped_text_mask, qs_text_boxes, relative_bboxes]

def crop_painting_for_text(imgs, bboxes):
    """ Rectangular paintings for images.
            inputs: imgs = [img1, img2,...]
            bboxes: text bboxes
        returns:
            [[painting1, painting2], [painting1], [painting1, painting2]]
    """
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
            x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
            largest_score = score

    mask = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    
    w_e, h_e = 0, 0 #int((w_box_1)*0.1), int((h_box_1)*0.3) # Extra height to account for errors

    pos = [x_box_1-w_e, y_box_1-h_e, x_box_1 + w_box_1 + w_e, y_box_1 + h_box_1 + h_e]
    mask[max(pos[1],0):min(pos[3], mask.shape[0]), max(pos[0],0):min(pos[2], mask.shape[1])] = 255
    return mask, pos

def text_detect_method1(img, opt = 0):
    """
    Text bounding box detection

    :param img: (ndarray) query image
    :param opt: (int) options to corner detection (0 or 1)
    :return: bbox: (tuple of int) bounding box, (tlx, tly, brx, bry)
    """

    bifilter = cv2.bilateralFilter(img, 9, 300, 300)

    hsv = cv2.cvtColor(bifilter, cv2.COLOR_BGR2HSV)

    ret, thresh = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    open1 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((10, 1), np.uint8), iterations=2)
    open2 = cv2.morphologyEx(open1, cv2.MORPH_OPEN, np.ones((1, 10), np.uint8), iterations=2)

    ret, labels = cv2.connectedComponents(open2)

    area = []
    for i, lab in enumerate(np.unique(labels)):
        area.append(open2[labels == lab].size)
    idx = sorted(range(len(area)), key=lambda k: area[k])

    x_n = open2.shape[0]
    nbb = list(range(int((x_n / 2) - x_n * (0.05)), int((x_n / 2) + x_n * (0.05))))

    if ret > 2:
        for i, j in enumerate(idx):
            if np.sum(open2[labels == j] == 0) == 0:
                idn = np.where(labels == j)[0]
                nocenter = [val for val in idn.tolist() if val in nbb]

                if (len(area) - i) > 2:
                    open2[labels == j] = 0

                if (len(area) - i) <= 2 and len(nocenter) > 0:
                    open2[labels == j] = 0
                    open2[labels == idx[i - 1]] = 255

    y0, x0, _, _, y1, x1, _, _ = detect_corners(open2)

    bbox = [x0, y0, x1, y1]
    mask = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255
    return mask, bbox

def detect_corners(mask):
    """
    Finds four points corresponding to rectangle corners

    :param mask: (ndarray) binary image
    :return: (int) points from corners
    """

    width = mask.shape[1]
    height = mask.shape[0]
    coords = np.argwhere(np.ones([height, width]))
    coords_x = coords[:, 1]
    coords_y = coords[:, 0]

    coords_x_filtered = np.extract(mask, coords_x)
    coords_y_filtered = np.extract(mask, coords_y)
    max_br = np.argmax(coords_x_filtered + coords_y_filtered)
    max_tr = np.argmax(coords_x_filtered - coords_y_filtered)
    max_tl = np.argmax(-coords_x_filtered - coords_y_filtered)
    max_bl = np.argmax(-coords_x_filtered + coords_y_filtered)

    tl_x, tl_y = int(coords_x_filtered[max_tl]), int(coords_y_filtered[max_tl])
    tr_x, tr_y = int(coords_x_filtered[max_tr]), int(coords_y_filtered[max_tr])
    bl_x, bl_y = int(coords_x_filtered[max_bl]), int(coords_y_filtered[max_bl])
    br_x, br_y = int(coords_x_filtered[max_br]), int(coords_y_filtered[max_br])

    return tl_x, tl_y, bl_x, bl_y, br_x, br_y, tr_x, tr_y

####
##

def compute_opening(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.erode(m, kernel, iterations=1) 
    m = cv2.dilate(m, kernel, iterations=1) 
    return m

def compute_closing(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.dilate(m, kernel, iterations=1) 
    m = cv2.erode(m, kernel, iterations=1) 
    return m

def brightText(img):
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    kernel = np.ones((1, int(img.shape[1] / 8)), np.uint8) 
    img_orig = cv2.dilate(img_orig, kernel, iterations=1) 
    img_orig = cv2.erode(img_orig, kernel, iterations=1) 
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)
        

def darkText(img):
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    kernel = np.ones((1, int(img.shape[1] / 8)), np.uint8) 
    img_orig = cv2.dilate(img_orig, kernel, iterations=1) 
    img_orig = cv2.erode(img_orig, kernel, iterations=1) 
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


def get_textbox_score(m, p_shape):
    m = m.copy()
    #plt.imshow(m)
    #plt.show()
    # we generate the minimum bounding box for the extracted mask
    x,y,w,h = cv2.boundingRect(m.astype(np.uint8))
    
    if w < 10 or h < 10 or h > w:
        return 0
    if w >= p_shape[0]*0.8 or h >= p_shape[1]/4:
        return 0

    # we compute the score according to its shape and its size
    sc_shape = np.sum(m[y:y+h, x:x+w]) / (w*h)
    sc_size = (w*h) / (m.shape[0] * m.shape[1])
    #sc_textboxish = 1 - (8 - w / h) ** 2 / 64
    final_score = (sc_shape + 50*sc_size) / 2
        
    return final_score

def get_best_textbox_candidate(mask, original_mask):
    x,y,w,h = cv2.boundingRect(original_mask.astype(np.uint8))
    p_shape = (w,h)
    p_coords = (x,y)
    
    mask_c = mask.copy()
    TH = 0.5
    i = 0
    found = False
    mask = None
    best_sc = 0
    while not found:
        biggest = extract_biggest_connected_component(mask_c).astype(np.uint8)
        if np.sum(biggest) == 0:
            return 0, None
        
        sc = get_textbox_score(biggest, p_shape)
        #print(f"{np.sum(biggest)} - {sc}")
        if sc > TH:
            #plt.imshow(biggest)
            #plt.show()
            mask = biggest
            best_sc = sc
            found = True
        else:
            mask_c -= biggest
            
    x, y, w, h = cv2.boundingRect(mask)
    M_W = 0.05
    M_H = 0.05
    ref = min(p_shape)
    x0,y0,x,y = (x - int(ref*M_W/2), y - int(ref*M_H/2), 
            (x+w) + int(ref*M_W/2), (y+h) + int(ref*M_H/2))
    return best_sc, [max(0,x0), max(0,y0), min(x, p_coords[0] + p_shape[0]), min(y, p_coords[1] + p_shape[1])]

def extract_biggest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the biggest connected component from a mask (0 and 1's).
    Args:
        img: 2D array of type np.float32 representing the mask
    
    Returns : 2D array, mask with 1 in the biggest component and 0 outside
    """
    # extract all connected components
    num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
    
    # we find and return only the biggest one
    max_val, max_idx = 0, -1
    for i in range(1, num_labels):
        area = np.sum(labels_im == i)
        if area > max_val:
            max_val = area
            max_idx = i
            
    return (labels_im == max_idx).astype(float)


def extract_paintings_from_mask(mask:np.ndarray):
    to_return = []
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    for lab in range(1, num_labels):      
        m = (labels == lab).astype(np.uint8)
        first_pixel = np.min(np.where(m != 0)[1])
        to_return.append((m, first_pixel))
    both = list(zip(*sorted(to_return, key=lambda t: t[1])))
    return both[0]

def extract_textbox(orig_img, mask=None):
    masks = []
    shapes = []
    bboxes = []
    if mask is None:
        masks = [np.ones(orig_img.shape[:2])]
    else:
        masks = extract_paintings_from_mask(mask)
        
    for m in masks:
        img = orig_img.copy()
        img[m == 0] = (0,0,0)

        sc_br, bbox_br = get_best_textbox_candidate(brightText(img), m)
        sc_dr, bbox_dr = get_best_textbox_candidate(darkText(img), m)
        bbox = bbox_br
        if sc_dr == 0 and sc_br == 0:
            continue
        if sc_dr > sc_br:
            bbox = bbox_dr
        bboxes.append(bbox)

    mask = np.zeros((orig_img.shape[0], orig_img.shape[1]))
    if len(bboxes) > 0:
        bboxes = bboxes[0]
    else:
        bboxes = [0,0,1,1]

    mask[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]] = 255
    cv2.imshow("mask", cv2.resize(mask,(512,512)))
    cv2.imshow("img", cv2.resize(orig_img,(512,512)))
    cv2.waitKey(0)
    return mask, bboxes

def generate_text_mask(shape, textboxes):
    if textboxes is None or len(textboxes) == 0:
        return np.zeros(shape).astype(np.uint8)
    
    mask = np.zeros(shape)
    for (xtl, ytl, xbr, ybr) in textboxes:
        pts = np.array(((xtl, ytl), (xtl, ybr), (xbr, ybr), (xbr, ytl)))
        cv2.fillConvexPoly(mask, pts, True)
    return mask.astype(np.uint8)


def eval_contours(contours, width):
        if len(contours) == 0: return 0
        if len(contours) == 1: return 0

        max_area = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            max_area.append(area)

        max_order = [0]
        for i in range(1, len(max_area)):
            for l in range(len(max_order)+1):
                if l == len(max_order):
                    max_order.append(i)
                    break
                elif max_area[i] > max_area[max_order[l]]:
                    max_order.insert(l, i)
                    break

        # Get the moments
        mu = [None] * len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        # Get the mass centers
        mc = [None] * len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i] ['m00'] + 1e-5))

        CM_order = [0]
        for i in range(1, len(mc)):

            for l in range(len(CM_order) + 1):
                if l == len(CM_order):
                    CM_order.append(i)
                    break
                elif abs(mc[i][0]-(width/2)) < abs(mc[CM_order[l]][0]-(width/2)):
                    CM_order.insert(l, i)
                    break

        return CM_order[0]

def text_detection_t4(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
    h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value
    #TextDetection.find_regions(img)
    # Open morphological transformation using a square kernel with dimensions 10x10
    kernel = np.ones((10, 10), np.uint8)
    s= cv2.GaussianBlur(s, (5, 5), 0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    # Convert the image to binary
    ret, th1 = cv2.threshold(morph_open, 35, 255, cv2.THRESH_BINARY_INV)

    # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
    shape = img.shape
    kernel = np.ones((shape[0] // 60, shape[1] // 4), np.uint8)
    th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    #th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

    # Find the contours
    (contours, hierarchy) = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        th3 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
        #th3 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
        (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the coordinates of the contours and draw it in the original image
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        c = contours[eval_contours(contours, shape[1])]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
        x = np.array([box[0][0],box[1][0],box[2][0],box[3][0]])
        y = np.array([box[0][1],box[1][1],box[2][1],box[3][1]])
        coordinates = np.array([min(x),min(y),max(x),max(y)])
        mask = np.zeros(th2.shape)
        mask[int(coordinates[1]-5):int(coordinates[3]+5), int(coordinates[0]-5):int(coordinates[2]+5)] = 255
    else:
        coordinates = np.zeros([4])
        mask = (np.ones(th2.shape)*255).astype(np.uint8)

    return mask, coordinates

def td5(img):
    im_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    im_y, _, _ = cv2.split(im_yuv)

    # Define kernel sizes
    kernel = np.ones((5, 10), np.float32)/9

    # Difference between erosion and dilation images
    y_dilation = cv2.morphologyEx(im_y, cv2.MORPH_DILATE, kernel, iterations=1)
    y_erosion = cv2.morphologyEx(im_y, cv2.MORPH_ERODE, kernel, iterations=1)

    difference_image = y_erosion - y_dilation

    # Grow contrast areas found
    growing_image = cv2.morphologyEx(difference_image, cv2.MORPH_ERODE, kernel, iterations=1)

    # Low pass filter to smooth out the result
    blurry_image = cv2.filter2D(growing_image, -1, kernel)

    # Thresholding the image to make a binary image
    ret, binary_image = cv2.threshold(blurry_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_binary_image = cv2.bitwise_not(binary_image)
    return inverted_binary_image
#
def td6(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,20))
    white = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    black = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    wgray = cv2.cvtColor(white,cv2.COLOR_BGR2GRAY)
    bgray = cv2.bitwise_not(cv2.cvtColor(black,cv2.COLOR_BGR2GRAY))
    _, white_mask = cv2.threshold(wgray, 200, 255, cv2.THRESH_TOZERO)
    _, black_mask = cv2.threshold(bgray, 200, 255, cv2.THRESH_TOZERO)
    white_masked = cv2.bitwise_and(gray, gray, mask=white_mask)
    black_masked = cv2.bitwise_and(inv, inv, mask=black_mask)
    _, white_img = cv2.threshold(white_masked, 200, 255, cv2.THRESH_TOZERO)
    _, black_img = cv2.threshold(black_masked, 200, 255, cv2.THRESH_TOZERO)
    return search_rectangles(white_img,black_img,img)

def search_rectangles(white,black,paint):
    white_paint = paint.copy()
    black_paint = paint.copy()
    step=10; start = 200; end = 256
    rectangles = []
    min_width = 150; min_height=20; max_height = 120
    detection_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,20))
    if paint.shape[0] > 1000:
        remove_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))
    else:
        remove_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    for v,value in enumerate(range(start,end,step)):
        white_found = cv2.inRange(white,value,value+step)
        black_found = cv2.inRange(black,value,value+step)
        white_found = cv2.morphologyEx(white_found,cv2.MORPH_CLOSE,detection_kernel)
        black_found = cv2.morphologyEx(black_found,cv2.MORPH_CLOSE,detection_kernel)
        white_found = cv2.morphologyEx(white_found,cv2.MORPH_OPEN,remove_kernel)
        black_found = cv2.morphologyEx(black_found,cv2.MORPH_OPEN,remove_kernel)
        if (np.sum(white_found,axis=None)!=0):
            white_canny = cv2.Canny(white_found,100,100,apertureSize=3)
            white_cont, _ = cv2.findContours(white_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(white_cont)==2:
                for cnt in white_cont:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if w > min_width and h > min_height and h < max_height:
                        cv2.rectangle(white_paint,(x,y),(x+w,y+h),(0,0,255),5)
                        rectangles.append((x,y,w,h))
            #cv2.imwrite('../results/TextBox/{0}_{1}_{2}.png'.format(k,p,'white'),white_paint)
        if (np.sum(black_found,axis=None)!=0):
            black_canny = cv2.Canny(black_found,100,100,apertureSize=3)
            black_cont, _ = cv2.findContours(black_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(black_cont)==2:
                for cnt in black_cont:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if w > min_width and h > min_height and h < max_height:
                        cv2.rectangle(black_paint,(x,y),(x+w,y+h),(0,0,255),5)
                        rectangles.append((x,y,w,h))
            #cv2.imwrite('../results/TextBox/{0}_{1}_{2}.png'.format(k,p,'black'),black_paint)
    mask = np.uint8(np.ones((paint.shape[0],paint.shape[1])))*255
    if rectangles:
        for rect in rectangles:
            x,y,w,h = rect
            mask[y:y+h,x:x+w] = 0
    else:
        x,y,w,h = 0,0,0,0

    return mask, [x,y,x+w,y+h]

# Selection utils
MM = "MM"
MM2 = "MM2"
MM3 = "MM3"
OPTIONS = [MM, MM2, MM3, "MM4"]

METHOD_MAPPING = {
    OPTIONS[0]: morphological_method1,
    OPTIONS[1]: best_segmentation,
    OPTIONS[2]: td6,
    OPTIONS[3]: text_detect_method1
}

def get_method(method):
    return METHOD_MAPPING[method]
