import numpy as np
import cv2
from background_mask import get_bbox
from debug_utils import *
from tqdm import tqdm
from collections import defaultdict

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

    print("Extracting text boxes...")
    for paintings in tqdm(cropped_imgs):
        for painting in paintings:
            result = text_segm(painting)
            cropped_text_mask.append(result[0].astype(bool))
            bboxes_mask.append(result[1])

    if isDebug():
        # Show images
        i = 0
        for paintings in cropped_imgs:
            for painting in paintings:
                bbox = bboxes_mask[i]
                painting_copy = painting.copy()
                painting_copy = cv2.rectangle(painting_copy, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0 ,0), 10)
                cv2.imshow("result text segm", cv2.resize(painting_copy,(512,512)))
                cv2.waitKey(0)
                i += 1

    return cropped_text_mask, bboxes_mask

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
    #cv2.imshow("text_mask", cv2.resize(polished_mask,(512,512)))
    #cv2.imshow("text_mask2", cv2.resize(polished_mask2,(512,512)))
    mask_best1, max_score1 = get_best_bbox(polished_mask)
    mask_best2, max_score2 = get_best_bbox(polished_mask2)
    cv2.imshow("text_mask_max", cv2.resize(mask_best1.astype(np.uint8)*255,(512,512)))
    cv2.imshow("text_mask_max2", cv2.resize(mask_best2.astype(np.uint8)*255,(512,512)))
    
    print(max_score1, max_score2)
    #cv2.waitKey(0)
    mask_best = mask_best1 if compute_score_mask(mask_best1.astype(np.uint8)*255) > compute_score_mask(mask_best2.astype(np.uint8)*255) else mask_best2
    return mask_best, get_bbox(mask_best.astype(np.uint8)*255)

def get_mask_ch(img, ch):
    img_s = get_channel(img, ch)
    img_c = img_s.copy().reshape(-1)
    img_c.sort()
    s_threshold = img_c[int(0.2*len(img_c))]
    mask = img_s < s_threshold

    tiny_region_letters = 0.02
    big_region_box = 0.05

    s1, s2 = int(img.shape[0]*0.01), int(img.shape[0]*0.01) #int(img.shape[0]*0.02), int(img.shape[1]*0.02)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, borderValue=0)

    s1, s2 = int(img.shape[0]*tiny_region_letters), int(img.shape[1]*tiny_region_letters)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_CLOSE, kernel, borderValue=0)

    s1, s2 = int(img.shape[0]*big_region_box), int(img.shape[0]*big_region_box*2) #int(img.shape[0]*big_region_box), int(img.shape[1]*big_region_box)
    kernel = np.ones((s1, s2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask*255, cv2.MORPH_OPEN, kernel, borderValue=0)
    
    positions = np.where(polished_mask==255)
    if len(positions[0]) != 0:
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
        score = compute_score_mask(mask_box)
        if score > max_score:
            max_score = score
            max_idx = idx

    return output == max_idx, max_score

def compute_score_mask(mask_box):
    try:
        x0,y0,x1,y1 = get_bbox(mask_box)
    except:
        return -1
    
    sum_pixels = (mask_box == 255).astype(np.uint8).sum()
    score = sum_pixels/((x1-x0)*(y1-y0))*(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))
    return score

def rere(img):
    return cv2.resize(img, (512,512))

def mask_morphological_ops(image):
    cv2.imshow("image", rere(image))
    im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    im_y, _, _ = cv2.split(im_yuv)
    im_y = im_y.astype(np.uint8)
    im_y_copy = im_y.copy()

    s1,s2 = 7,7
    kernel = np.ones((s1, s2), np.float32)
    im_y = cv2.morphologyEx(im_y, cv2.MORPH_ERODE, kernel, iterations = 1)

    s1,s2 = 5,5
    kernel = np.ones((s1, s2), np.float32)
    # Difference between erosion and dilation images 
    y_dilation = cv2.morphologyEx(im_y, cv2.MORPH_DILATE, kernel, iterations = 1)
    y_erosion = cv2.morphologyEx(im_y, cv2.MORPH_ERODE, kernel, iterations = 1)

    s1, s2 = 4,4 
    kernel = np.ones((s1, s2), np.float32)

    difference_image = y_dilation - y_erosion
    difference_image = cv2.morphologyEx(difference_image, cv2.MORPH_ERODE, kernel, iterations = 1)
    text_mask = (difference_image < 3).astype(np.uint8)*255
    cv2.imshow("diff img", rere(difference_image))
    
    # Finding contours of the white areas of the images (high possibility of text)
    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    text_mask = cv2.fillPoly(text_mask, contours, 255).astype(np.uint8)
    cv2.imshow("text mask", rere(text_mask))

    # Initialize parameters
    largest_area, second_largest_area, x_box_1, y_box_1, w_box_1, h_box_1, x_box_2, y_box_2, w_box_2, h_box_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    max_score1, max_score2 = -1, -1
    image_width = text_mask.shape[0]

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        x0,y0,x1,y1 = x, y, x+w, y+h 
        #area = cv2.contourArea(cnt)
        
        score = compute_score_mask_text(text_mask, [x0,y0,x1,y1])
        if ((w/h > 2) & (w/h < 12) & (w > (0.1 * image_width))) and score > max_score2:

            if score > max_score1:
                x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
                x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                max_score2 = max_score1
                max_score1 = score

            else:
                x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                max_score2 = score

    a = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
    a[y_box_1:y_box_1+h_box_1, x_box_1:x_box_1+w_box_1] = 255
    b = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
    b[y_box_2:y_box_2+h_box_2, x_box_2:x_box_2+w_box_2] = 255

    #cv2.imshow("big1", rere(a))
    #cv2.imshow("big2", rere(a))
    #cv2.waitKey(0)

    return a==255, [x_box_1, y_box_1, x_box_1+w_box_1, y_box_1 + h_box_1]

def morphological_method3(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,0]
    s1,s2 = img.shape[0]//25, img.shape[1]//25

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)))
    tophat = closing - opening
    #blur = cv2.GaussianBlur(tophat, (7, 7), 0)

    #thresh = tophat > 0.5*tophat.max()
    copy_th = tophat.copy().reshape(-1)
    copy_th.sort()
    thresh = tophat > copy_th[int(0.95*len(copy_th))] #0.7*tophat.max() #cv2.threshold(tophat, 0.5*tophat.max(), 255, cv2.THRESH_BINARY)[1]

    tophat = closing - opening

    current_threshold = 0.8

    thresh = tophat > 0.7*tophat.max() 
    #text_mask = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[0]*0.2), 5)))
    #text_mask = text_mask*255

    dilation = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[0]*0.15), 3)))
    s_v = int(img.shape[1]*0.01)
    text_mask = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, s_v)))
    cv2.imshow("test_mask", cv2.resize(text_mask*255,(512,512)))
    # Finding contours of the white areas of the images (high possibility of text)
    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # Initialize parameters
    largest_area, second_largest_area, x_box_1, y_box_1, w_box_1, h_box_1, x_box_2, y_box_2, w_box_2, h_box_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    max_score1, max_score2 = -1, -1
    image_width = text_mask.shape[0]

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        x0,y0,x1,y1 = x, y, x+w, y+h 
        #area = cv2.contourArea(cnt)
        # ((w/h > 2) & (w/h < 12) & (w > (0.1 * image_width))) and
        score = compute_score_mask_text(text_mask, [x0,y0,x1,y1])

        if score > max_score2 and w > 0.1*text_mask.shape[0]:

            if score > max_score1:
                x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
                x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                max_score2 = max_score1
                max_score1 = score

            else:
                x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                max_score2 = score

    a = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    a[y_box_1:y_box_1+h_box_1, x_box_1:x_box_1+w_box_1] = 255
    #cv2.imshow("img", cv2.resize(img,(512,512)))
    #cv2.imshow("a", cv2.resize(a,(512,512)))
    #cv2.waitKey(0)
    return a, [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]

def morphological_method4(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s1,s2 = img.shape[0]//20, img.shape[1]//20

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)), borderValue=0)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)), borderValue=0)
    tophat = closing - opening
    #blur = cv2.GaussianBlur(tophat, (7, 7), 0)

    m1, m2 = int(0.05*img.shape[0]), int(0.05*img.shape[1])
    tophat = tophat[m1:img.shape[0]-m1, m2:img.shape[1]-m2]

    #thresh = tophat > 0.5*tophat.max()
    alpha = 0.7
    copy_th = tophat.copy().reshape(-1)
    copy_th.sort()
    thresh = tophat > copy_th[int(0.99*len(copy_th))] #0.7*tophat.max() #cv2.threshold(tophat, 0.5*tophat.max(), 255, cv2.THRESH_BINARY)[1]

    #m1, m2 = int(0.02*thresh.shape[0]), int(0.02*thresh.shape[1])
    #thresh[m1:thresh.shape[0]-m1, m2:thresh.shape[1]-m2] = False

    contours, _ = cv2.findContours(thresh.astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    """ cv2.imshow("prefilter", cv2.resize(thresh.astype(np.uint8)*255, (512,512)))
    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
    
        dev1 = (closing[y:y+h,x:x+w]).std() < 40
        dev2 = (opening[y:y+h,x:x+w]).std() < 40
        print((closing[y:y+h,x:x+w]).std(), (opening[y:y+h,x:x+w]).std())
        if not dev1 and not dev2:
            print("a")
            thresh[y:y+h,x:x+w] = False

    """
    cv2.imshow("postfilter", cv2.resize(thresh.astype(np.uint8)*255, (512,512)))
    #while thresh.astype(np.uint8).sum() > 

    #filled = fill_holes(thresh)
    #thresh = cv2.threshold(thresh4,250,255,cv2.THRESH_BINARY)[1]
    #imshow(thresh)

    #dilation = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    #expansion = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    #dilation = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[0]*0.4), 1)), borderValue=0)
    s_v = int(img.shape[1]*0.02)
    padding = 250
    dilation = cv2.copyMakeBorder(src=thresh.astype(np.uint8)*255, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0) 
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[0]*0.4), 1)), borderValue=0)
    cv2.imshow("pro0", cv2.resize(dilation.astype(np.uint8), (512,512)))
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[0]*0.4), 1)), borderValue=0)
    
    dilation = dilation[padding:dilation.shape[0]-padding, padding:dilation.shape[1]-padding]
    cv2.imshow("pro1", cv2.resize(dilation.astype(np.uint8), (512,512)))
    
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (s_v*2, 1)), borderValue=0)
    #dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (, 1)))
    s_v = int(img.shape[1]*0.01)
    text_mask = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, s_v)))
    cv2.imshow("pro2", cv2.resize(text_mask.astype(np.uint8), (512,512)))

    # Finding contours of the white areas of the images (high possibility of text)
    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # Initialize parameters
    largest_area, second_largest_area, x_box_1, y_box_1, w_box_1, h_box_1, x_box_2, y_box_2, w_box_2, h_box_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    max_score1, max_score2 = -1, -1
    image_width = text_mask.shape[0]
    cv2.imshow("pretakeboxes", cv2.resize(text_mask.astype(np.uint8), (512,512)))

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        x0,y0,x1,y1 = x, y, x+w, y+h 
        #area = cv2.contourArea(cnt)
        # ((w/h > 2) & (w/h < 12) & (w > (0.1 * image_width))) and
        score = compute_score_mask_text(text_mask, [x0,y0,x1,y1])

        if score > max_score2 and w > 0.1*text_mask.shape[1] and (w/h > 1.5) and (w/h < 40):

            if score > max_score1:
                x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
                x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                max_score2 = max_score1
                max_score1 = score

            else:
                x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                max_score2 = score

    a = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    w_e, h_e = int((w_box_1)*0.05), int((h_box_1)*0.5)

    pos = [m1+x_box_1, m2+y_box_1-h_e, x_box_1 + m1 + w_box_1, m2 + y_box_1 + h_box_1+h_e]
    a[max(pos[1],0):min(pos[3], a.shape[0]), max(pos[0],0):min(pos[2], a.shape[1])] = 255
    cv2.imshow("img", cv2.resize(img,(512,512)))
    cv2.imshow("a", cv2.resize(a,(512,512)))
    cv2.waitKey(0)
    return a, pos

def test2(image):
    im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    im_y, _, _ = cv2.split(im_yuv)

    # Define kernel sizes
    kernel = np.ones((5, 5), np.float32)/9

    # Difference between erosion and dilation images 
    y_dilation = cv2.morphologyEx(im_y, cv2.MORPH_DILATE, kernel, iterations = 1)
    y_erosion = cv2.morphologyEx(im_y, cv2.MORPH_ERODE, kernel, iterations = 1)

    difference_image = y_erosion - y_dilation

    # Grow contrast areas found
    growing_image = cv2.morphologyEx(difference_image, cv2.MORPH_ERODE, kernel, iterations = 1)

    # Low pass filter to smooth out the result
    blurry_image = cv2.filter2D(growing_image, -1, kernel)

    # Thresholding the image to make a binary image
    ret, binary_image = cv2.threshold(blurry_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
    inverted_binary_image = cv2.bitwise_not(binary_image)

    # Clean small white pixels areas outside text using closing filter
    #text_mask = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_OPEN, kernel, iterations = 1)

    text_mask = inverted_binary_image
    
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

    a = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
    a[y_box_1:y_box_1+h_box_1, x_box_1:x_box_1+w_box_1] = 255
    cv2.imshow("img", rere(image))
    cv2.imshow("res", rere(a))
    cv2.waitKey(0)
    return a, [y_box_1, y_box_1+h_box_1, x_box_1, x_box_1 + w_box_1]

def compute_score_mask_text(mask_box, bbox):
    x0,y0,x1,y1 = bbox
    mask_box_part = mask_box[y0:y1,x0:x1]
    sum_pixels = (mask_box_part == 255).astype(np.uint8).sum()
    md = max(mask_box.shape[0], mask_box.shape[1])
    score = (1-abs(0.3-(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))))+\
            2*((mask_box.shape[0]*mask_box.shape[1])/(md*md))
    return score

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

############
import numpy as np
import imutils
import cv2
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_boxes(image, boxes, color=(0, 255, 0)):
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        #print('({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(x1, y1, x2, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
    return image


def find_text_region():
    with open("../w5_text_bbox_list.pkl", "rb") as fp:
        # tlx, tly, brx, bry
        bbox_gt = pickle.load(fp)

    top_limit_vector = []
    bottom_limit_vector = []
    area_vector = []
    for image in glob.glob('../data/w5_BBDD_random/*.jpg'):
        im = cv2.imread(image)
        index = int(os.path.split(image)[-1].split(".")[0].split("_")[1])
        tlx, tly, brx, bry = bbox_gt[index]
        H, W, _ = np.shape(im)
        h = bry - tly
        w = brx - tlx
        area_vector.append((h * w) / (H * W))
        if bry < H / 2:
            top_limit_vector.append(bry / H)
        else:
            if tly / H > 1:
                print(image)
                print(bbox_gt[index])
            bottom_limit_vector.append(tly / H)
    top_limit = max(top_limit_vector)
    bottom_limit = min(bottom_limit_vector)
    print("Top and bottom limits", top_limit, bottom_limit)
    print("Min and max areas", min(area_vector), max(area_vector))


def fill_holes(mask):
    im_floodfill = mask.astype(np.uint8).copy()
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 1)
    return mask.astype(np.uint8) | cv2.bitwise_not(im_floodfill)


def merge_boxes(boxes):
    y = np.array([[(b[1] + b[3]) / 2, b[3] - b[1]] for b in boxes])
    clt = DBSCAN(eps=22, min_samples=1, metric='l1').fit(y)
    labels = clt.labels_

    clusters = defaultdict(list)
    for box, label in zip(boxes, labels):
        if label != -1:
            clusters[label].append(box)
    clusters = clusters.values()

    merged_boxes = []
    areas = []
    for clt in clusters:
        num_pixel_estimation = 20
        
        positions = np.where(mask==255)
        hs, ws = sorted(positions[0]), sorted(positions[1])
        h_min, h_max = int(np.array(hs[:num_pixel_estimation]).mean()), int(np.array(hs[-num_pixel_estimation:]).mean())
        w_min, w_max = int(np.array(ws[:num_pixel_estimation]).mean()), int(np.array(ws[-num_pixel_estimation:]).mean())
        x, y, w, h = w_min, h_min, w_max-w_min, h_max-h_min #cv2.boundingRect(points=np.concatenate(clt).reshape(-1, 2))
        merged_boxes.append((x, y, x + w, y + h))

        area = np.sum([(b[2]-b[0])*(b[3]-b[1]) for b in clt])
        areas.append(area)

    return merged_boxes, areas


def detect(img, method='difference', show=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_h, im_w = img.shape[:2]

    def tophat(img):
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        blackhat = cv2.cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        tophat = tophat if np.sum(tophat) > np.sum(blackhat) else blackhat
        if show:
            imshow(tophat)

        thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if show:
            imshow(thresh)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        return thresh

    def difference(img):
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3)))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        tophat = closing - opening
        #blur = cv2.GaussianBlur(tophat, (7, 7), 0)

        #thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.threshold(tophat, 0.5*tophat.max(), 255, cv2.THRESH_BINARY)[1]

        #filled = fill_holes(thresh)
        #thresh = cv2.threshold(thresh4,250,255,cv2.THRESH_BINARY)[1]
        #imshow(thresh)

        dilation = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
        #expansion = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

        if show:
            imshow(closing)
            imshow(opening)
            imshow(tophat)
            imshow(thresh)

        return dilation

    func = {
        'tophat': tophat,
        'difference': difference
    }

    # find contours
    mask = func[method](img)
    if show:
        imshow(mask)

    # detect boxes from contours
    contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]
    boxes, bad_boxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        image_area = im_h * im_w
        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = area / rect_area

        # filter boxes
        cond1 = extent > 0.2
        cond2 = h > 10
        cond3 = (rect_area / image_area) <= 0.2935
        cond4 = (w / h) > 1
        cond5 = (y / im_h) >= 0.5719 or ((y + h) / im_h) <= 0.2974
        #print(cond1, cond2, cond3, cond4, cond5)

        if all([cond1, cond2, cond3, cond4, cond5]):
            boxes.append((x, y, x + w, y + h))
        else:
            bad_boxes.append((x, y, x + w, y + h))
    if show:
        tmp = draw_boxes(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), bad_boxes, color=(255, 0, 0))
        tmp = draw_boxes(tmp, boxes, color=(0, 255, 0))
        imshow(tmp)

    # merge boxes
    if boxes:
        merged_boxes, areas = merge_boxes(boxes)
        if show:
            imshow(draw_boxes(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), merged_boxes))

        filtered_boxes = []
        filtered_areas = []
        for box, area in zip(merged_boxes, areas):
            tlx, tly, brx, bry = box
            h = bry-tly
            w = brx-tlx
            if 0.05 < h/w < 0.25:
                filtered_boxes.append(box)
                filtered_areas.append(area)
        if filtered_boxes:
            idx = np.argmax(filtered_areas)
            boxes = [filtered_boxes[idx]]
        else:
            boxes = []

    return boxes


def correct_boxes(boxes, orig_h, orig_w, h, w):
    w_ratio = orig_w / w
    h_ratio = orig_h / h

    corrected = []
    for b in boxes:
        tlx = int(np.floor(b[0] * w_ratio))
        tly = int(np.floor(b[1] * h_ratio))
        brx = int(np.ceil(b[2] * w_ratio))
        bry = int(np.ceil(b[3] * h_ratio))
        corrected.append((tlx, tly, brx, bry))

    return corrected


def filter_text_keypoints(img, keypoints):
    resized = imutils.resize(img, width=512)
    boxes = detect(resized)
    boxes = correct_boxes(boxes, *img.shape[:2], *resized.shape[:2])

    def inside(pt, box):
        # point = (x, y)
        # box = (tlx, tly, brx, bry)
        return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

    filtered = []
    for kp in keypoints:
        for box in boxes:
            if inside(kp.pt, box):
                break
        else:
            filtered.append(kp)

    return filtered


def compute_text_mask(img, method='difference'):
    resized = imutils.resize(img, width=512)
    boxes = detect(resized, method)
    boxes = correct_boxes(boxes, *img.shape[:2], *resized.shape[:2])

    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    for box in boxes:
        tlx, tly, brx, bry = box
        mask[tly:bry, tlx:brx] = 0

    return mask


#

# Selection utils
SATURATION_MASKING = "SM"
MM = "MM"
D = "DUMMY"
OPTIONS = [SATURATION_MASKING, MM, D]

METHOD_MAPPING = {
    OPTIONS[0]: saturation_masking,
    OPTIONS[1]: morphological_method4,
    OPTIONS[2]: detect
}

def get_method(method):
    return METHOD_MAPPING[method]
