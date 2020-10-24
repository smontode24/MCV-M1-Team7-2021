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

    cv2.imshow("t1", cv2.resize(tophat*255,(512,512)))
    cv2.imshow("t2", cv2.resize(thresh.astype(np.uint8)*255,(512,512)))
    
    current_threshold = 0.8

    #thresh = tophat > 0.7*tophat.max() 
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

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)), borderType=cv2.BORDER_REPLICATE)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s1, s2)), borderType=cv2.BORDER_REPLICATE)
    tophat = closing - opening
    #blur = cv2.GaussianBlur(tophat, (7, 7), 0)

    m1, m2 = int(0.05*img.shape[0]), int(0.05*img.shape[1])
    tophat = tophat[m1:img.shape[0]-m1, m2:img.shape[1]-m2]

    #thresh = tophat > 0.5*tophat.max()
    thr = 0.95
    copy_th = tophat.copy().reshape(-1)
    copy_th.sort()
    thresh = tophat > copy_th[int(0.98*len(copy_th))] # thr*tophat.max() ##0.7*tophat.max() #cv2.threshold(tophat, 0.5*tophat.max(), 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), borderValue=0)

    cv2.imshow("t1", cv2.resize(tophat,(512,512)))

    #m1, m2 = int(0.02*thresh.shape[0]), int(0.02*thresh.shape[1])
    #thresh[m1:thresh.shape[0]-m1, m2:thresh.shape[1]-m2] = False

    contours, _ = cv2.findContours(thresh.astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.imshow("prefilter", cv2.resize(thresh.astype(np.uint8)*255, (512,512)))
    #cv2.imshow("prehat", cv2.resize(tophat,(512,512)))
    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
    
        dev1 = (closing[y:y+h,x:x+w]).std() < 20
        dev2 = (opening[y:y+h,x:x+w]).std() < 20
        #print((closing[y:y+h,x:x+w]).std(), (opening[y:y+h,x:x+w]).std())
        area = thresh[y:y+h, x:x+w].astype(np.uint8).sum()/(h*w)
        #if area < 0.3: #or (not dev1 and not dev2):
        #    #print("a")
        #    thresh[y:y+h, x:x+w] = False
        #    #tophat[y:y+h,x:x+w] = 0
        
    #th_m1,th_m2 = int(0.025*tophat.shape[0]), int(0.1*tophat.shape[1])
    ##tophat[:m1, :] = 0
    ##tophat[tophat.shape[0]-m1:, :] = 0
    #tophat[:, :th_m2] = 0
    #tophat[:, tophat.shape[1]-th_m2:] = 0

    #copy_th = tophat.copy().reshape(-1)
    #copy_th.sort()
    #thresh = tophat > thr*tophat.max() #copy_th[int(0.99*len(copy_th))] #0.7*tophat.max() #cv2.threshold(tophat, 0.5*tophat.max(), 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh.astype(np.uint8)*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,3)))
    #thresh = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, )), borderValue=0)
    #ret, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

    #cv2.imshow("post_hat", cv2.resize(tophat, (512,512)))
    #
    cv2.imshow("postfilter", cv2.resize(thresh, (512,512)))
    #while thresh.astype(np.uint8).sum() > 

    #filled = fill_holes(thresh)
    #thresh = cv2.threshold(thresh4,250,255,cv2.THRESH_BINARY)[1]
    #imshow(thresh)

    #dilation = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    #expansion = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    #dilation = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[0]*0.4), 1)), borderValue=0)
    s_v = int(img.shape[1]*0.02)
    padding = 250 # .astype(np.uint8)*255
    dilation = cv2.copyMakeBorder(src=thresh, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0) 
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[1]*0.6), 11)), borderValue=0)
    #cv2.imshow("pro0", cv2.resize(dilation.astype(np.uint8), (512,512)))
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[1]*0.6), 7)), borderValue=0)
    cv2.imshow("pro02", cv2.resize(dilation.astype(np.uint8), (512,512)))

    dilation = cv2.morphologyEx(dilation, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[1]*0.6), 7)), borderValue=0)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(img.shape[1]*0.6), 7)), borderValue=0)
    
    text_mask = dilation[padding:dilation.shape[0]-padding, padding:dilation.shape[1]-padding]
    cv2.imshow("pro1", cv2.resize(dilation.astype(np.uint8), (512,512)))
    
    #dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (s_v*2, 1)), borderValue=0)
    ##dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (, 1)))
    #s_v = int(img.shape[1]*0.01)
    #text_mask = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, s_v)))
    #cv2.imshow("pro2", cv2.resize(text_mask.astype(np.uint8), (512,512)))

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
    w_e, h_e = int((w_box_1)*0.01), int((h_box_1)*0.4)

    pos = [m1+x_box_1-w_e, m2+y_box_1-h_e, x_box_1 + m1 + w_box_1 + w_e, m2 + y_box_1 + h_box_1+h_e]
    a[max(pos[1],0):min(pos[3], a.shape[0]), max(pos[0],0):min(pos[2], a.shape[1])] = 255
    cv2.imshow("img", cv2.resize(img,(512,512)))
    cv2.imshow("result", cv2.resize(a,(512,512)))
    cv2.waitKey(0)
    return a, pos

def text_detect_method1(img, opt = 1):
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

    # plt.figure()
    # plt.imshow(open2, cmap = 'gray')

    ret, labels = cv2.connectedComponents(open2)
    # plt.figure()
    # plt.imshow(labels, cmap = 'gray')

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

    if opt == 0:

        # Coordinates of non-black pixels.
        coords = np.argwhere(open2)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1

    else:
        y0, x0, _, _, y1, x1, _, _ = detect_corners(open2)

    bbox = (x0, y0, x1, y1)
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask[y0:y1,x0:x1] = 255
    return mask, bbox


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
def text_detection(img):
    """
    This function detects the text in the image and returns an array with coordinates of text bbox.
        input: image in BGR spacecolor.
        output: [tlx1, tly1, brx1, bry1] where t = top, b = bottom, l = left, r = right
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
    h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value

    # Open morphological transformation using a square kernel with dimensions 10x10
    kernel = np.ones((10, 10), np.uint8)
    morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    # Convert the image to binary
    ret, th1 = cv2.threshold(morph_open, 30, 255, cv2.THRESH_BINARY_INV)

    # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
    shape = img.shape
    kernel = np.ones((shape[0] // 50, shape[1] // 5), np.uint8)
    th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    # Find the contours
    (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the coordinates of the contours and draw it in the original image
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
        coordinates = np.concatenate([box[0], box[2]])
    else:
        coordinates = np.zeros([4])
    
    return [], coordinates


def bounding_boxes_detection(image):
    """
    This function detects the bounding boxes of the text in all the images of a specific folder
    :param image_path: path of the images
    :param mask_set_path: path where the masks will be saved
    :param method: 1 for color segmentation and 2 for morphology operations
    :param save_masks: bool indicating if the masks need to be saved
    :param idx: int containing the index of the image
    :return: list of bounding boxes from first image to last image. Each image contains a maximum of 2 bounding boxes.
        [[[first_bounding_box_of_first_image],[second_bounding_box_of_second_image]], [[first_bounding_box_of_second_image]], ...]
    Each bounding box has the following int values:
        [lowest_pixel_x, lowest_pixel_y, highest_pixel_x, highest_pixel_y] 
    
    """
    method=2
    # Create the empty list to store the bounding boxes coordinates
    boxes = []

    # Define grayscale image
    im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    im_y, _, _ = cv2.split(im_yuv)

    # Define kernel sizes
    kernel = np.ones((3, 3), np.float32)

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

    # Clean small white pixels areas outside text using closing filter
    #text_mask = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_OPEN, kernel, iterations = 1)

    text_mask = inverted_binary_image

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

        if (w / h > 2) & (w / h < 12) & (w > (0.1 * image_width)) & (area > second_largest_area):

            if area > largest_area:
                x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
                x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                second_largest_area = largest_area
                largest_area = area

            else:
                x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                second_largest_area = area

    # cv2.rectangle(image, (x_box_1, y_box_1), (x_box_1 + w_box_1 - 1, y_box_1 + h_box_1 - 1), 255, 2)
    # cv2.rectangle(image, (x_box_2, y_box_2), (x_box_2 + w_box_2 - 1, y_box_2 + h_box_2 - 1), 255, 2)

    # Append the corners of the bounding boxes to the boxes list

    
    return [],[x_box_1, y_box_1, x_box_1+w_box_1, y_box_1+h_box_1]



def compute_score_mask_text(mask_box, bbox):
    x0,y0,x1,y1 = bbox
    mask_box_part = mask_box[y0:y1,x0:x1]
    sum_pixels = (mask_box_part == 255).astype(np.uint8).sum()
    md = max(mask_box.shape[0], mask_box.shape[1])
    score = (1-abs(0.3-(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))))+\
            ((mask_box.shape[0]*mask_box.shape[1])/(md*md))+0.2*sum_pixels/((y1-y0)*(x1-x0))
    return score

def text_detection2(img):
    """
    This function detects the text in the image and returns an array with coordinates of text bbox.
    input: image in BGR spacecolor.
    output: [min x, min y, max x, max y]
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
    h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value

    # Open morphological transformation using a square kernel with dimensions 10x10
    kernel = np.ones((15, 15), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    # Convert the image to binary
    ret, th1 = cv2.threshold(morph_open, 20, 255, cv2.THRESH_BINARY_INV)

    # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
    shape = img.shape
    kernel = np.ones((shape[0] // 60, shape[1] // 4), np.uint8)
    th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

    # Find the contours
    (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize parameters

    # Find the coordinates of the contours and draw it in the original image
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
        x = np.array([box[0][0],box[1][0],box[2][0],box[3][0]])
        y = np.array([box[0][1],box[1][1],box[2][1],box[3][1]])
        coordinates = np.array([min(x),min(y),max(x),max(y)])
    else:
        coordinates = np.zeros([4]).astype(np.uint8)
       
      
    return [], coordinates

def compute_score_mask2(mask_box, bbox):
    x0,y0,x1,y1 = bbox
    sum_pixels = (mask_box[y0:y1, x0:x1] == 255).astype(np.uint8).sum()
    
    mask_box_cp = mask_box.copy()
    mean_x = int(np.where(mask_box_cp==255)[1].mean()) 
    score = 0.2*sum_pixels/((x1-x0)*(y1-y0))+\
        0.8*(sum_pixels/(mask_box.shape[0]*mask_box.shape[1]))+\
        1*abs((mask_box.shape[1]//2)-mean_x)/(mask_box.shape[1]//2) # TODO: Check correctness
    return score


def best_segmentation(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
    h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value

    # Open morphological transformation using a square kernel with dimensions 10x10
    s = cv2.medianBlur(s, 5)
    kernel = np.ones((1,30)) 
    morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    # Convert the image to binary
    morph_cp = morph_open.copy()
    morph_cp = morph_cp.reshape(-1)
    morph_cp.sort()
    th1 = morph_open < morph_cp[int(len(morph_cp)*0.075)]+5 #TODO: tune 5 
    
    th_m1,th_m2 = int(0.025*th1.shape[0]), int(0.1*th1.shape[1])
    th1[:th_m1, :] = False
    th1[th1.shape[0]-th_m1:, :] = False
    th1[:, :th_m2] = False
    th1[:, th1.shape[1]-th_m2:] = False

    th1 = th1.astype(np.uint8)*255
    #cv2.imshow("th1", rere(th1))

    # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
    padding = max(img.shape[0], img.shape[1])
    th1 = cv2.copyMakeBorder(src=th1, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=0) 
    
    kernel = np.ones((7, 1), np.uint8)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, img.shape[1] // 16), np.uint8) # Increase
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((img.shape[0] // 64, img.shape[1] // 4), np.uint8)
    th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    th3 = th2 
    th3 = th3[padding:-padding, padding:-padding]
    #cv2.imshow("th1", rere(th1[padding:-padding, padding:-padding]))
    #cv2.imshow("th2", rere(th2[padding:-padding, padding:-padding]))
    #cv2.imshow("th3", rere(th3))
    #cv2.imshow("final proposals",rere(th3))
    text_mask = th3

    # Find the contours
    (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # compute_score_mask2

    contours, _ = cv2.findContours(text_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Initialize parameters
    largest_score, x_box_1, y_box_1, w_box_1, h_box_1 = 0, 0, 0, 0, 0
    image_width = text_mask.shape[0]

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        x0,y0,x1,y1 = x, y, x+w, y+h 
        score = compute_score_mask2(text_mask, [x0,y0,x1,y1])
        if (w / h > 2) & (w / h < 30) & (w > (0.1 * image_width)) and score > largest_score:
            x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
            x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
            largest_score = score

    a = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    pos = [x_box_1, y_box_1, x_box_1 + w_box_1 , y_box_1 + h_box_1]
    a[max(pos[1],0):min(pos[3], a.shape[0]), max(pos[0],0):min(pos[2], a.shape[1])] = 255
    #cv2.imshow("img", cv2.resize(img,(512,512)))
    #cv2.imshow("result", cv2.resize(a,(512,512)))
    #cv2.waitKey(0)
    return a, pos

# Selection utils
MM = "MM"
MM2 = "MM2"
OPTIONS = [MM, MM2]

METHOD_MAPPING = {
    OPTIONS[0]: morphological_method4,
    OPTIONS[1]: best_segmentation
}

def get_method(method):
    return METHOD_MAPPING[method]
