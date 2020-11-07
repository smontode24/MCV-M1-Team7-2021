import numpy as np
import cv2
from metrics import *
import scipy.stats as stats
from debug_utils import *
from evaluation.mask_evaluation import bb_intersection_over_union
from tqdm import tqdm
from scipy import ndimage

def bg_mask(query_imgs, method): 
    """ Obtain a mask for each image in the list of images query_img. The method will be determined by the passed method argument.
    Available methods are: "PBM". 
        params: 
            query_imgs: List of images of the query set
            method: ["PBM"]
        returns:
        Returns 3 elements:
            - List of masks [2D images with 1 channel]. A pixel = 0 means background, and pixel = 255 painting
            - Bboxes: [[[3,3,7,7], [3,2,70,70]], [[2,2,60,60]]]
            - Results_splitted_segm: [[2d img, 2dimg], [2dimg], ...] each 2d image contains the segmentation for one single painting
    """
    print("Obtaining masks")
    segmentation_method = get_method(method)

    results_segmentation, results_bboxes, results_splitted_segmentation = [], [], []
    for img in tqdm(query_imgs):
        segm, bbox, sep_masks = segmentation_method(img)
        results_segmentation.append(segm)
        results_bboxes.append(bbox)
        results_splitted_segmentation.append(sep_masks)

    if isDebug():
        i = 0
        for mask in results_segmentation:
            
            img_copy = np.zeros_like(query_imgs[i])
            img_copy[mask==255] = query_imgs[i][mask==255]
            img_copy = cv2.resize(img_copy, (512, 512))
            cv2.imshow("debug", img_copy)
            cv2.waitKey(0)
            i += 1

    return [results_segmentation, results_bboxes, results_splitted_segmentation]

def apply_mask(query_imgs, masks, method): 
    """ Apply mask to each image in the query set, based on the method that was applied 
            params:
                query_imgs: List of images that will be masked
                masks: List of masks to apply to each image
        returns: 
                List of masked images
    """
    resulting_imgs = []
    for img, mask in zip(query_imgs, masks):
        positions = np.where(mask == 255)
        if method == CBHS: # Special treatment for cell-based bg segmentation to mantain 
            x_min, x_max, y_min, y_max = positions[0][0], positions[0][-1], positions[1][0], positions[1][-1]
            img = img[x_min:x_max, y_min:y_max]
        else:
            mask = mask == 255
            img = img[mask].reshape(-1, 3)

        resulting_imgs.append(img)
        
        if isDebug():
            addDebugImage(img)
    if isDebug():
        showDebugImage()
        print("Finished to apply masks")
    
    return resulting_imgs

def edge_segmentation(img):
    """ Detect edges to create a mask that indicates where the paintings are located """
    sx, sy = np.shape(img)[:2]
    datatype = np.uint8

    kernel = np.ones((7,7), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(img, 25, 30)
    
    # Closing to ensure edges are continuous
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Filling
    kernel = np.ones((11,11), dtype=np.uint8)
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

    bboxes = [get_bbox(bc)]
    resulting_masks = bc
    splitted_resulting_masks = [bc]
    
    # Second painting if first one does not take most part + more or less a rectangular shape + no IoU
    if len(idxs) > 1:
        if not takes_most_part_image(bc) and regular_shape(sbc) and check_no_iou(bc, sbc):
            bboxes.append(get_bbox(sbc))
            resulting_masks = np.logical_or(resulting_masks==255, sbc==255).astype(np.uint8)*255
            splitted_resulting_masks.append(sbc)

            # Third painting
            if len(idxs) > 2:
                if regular_shape(tbc) and check_no_iou(bc, tbc) and check_no_iou(sbc, tbc):
                    bboxes.append(get_bbox(tbc))
                    resulting_masks = np.logical_or(resulting_masks==255, tbc==255).astype(np.uint8)*255
                    splitted_resulting_masks.append(tbc)
    
    return resulting_masks, bboxes, splitted_resulting_masks

def refine_mask(img, mask, bbox):
    original_mask = mask.copy()
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    seeds = [(0, mask.shape[0]//2), (mask.shape[1]-5, mask.shape[0]//2), (mask.shape[1]//2, 0), (mask.shape[1]//2, mask.shape[0]-5)]
    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    best_and_lowest = 100
    best_mask = None

    for seed in seeds:
        floodflags = 4
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        floodflags |= (255 << 8)

        mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
        num,img,mask,rect = cv2.floodFill(img, mask, seed, (255,0,0), (5,)*3, (5,)*3, floodflags)
        #cv2.imshow("m1", mask)
        #cv2.waitKey(0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11,11)))

        if mask.astype(np.uint8).sum() != 0:
            mask_bbox = get_bbox((mask!=0).astype(np.uint8)*255)
            sum_pixels = (mask!=0).astype(np.uint8).sum()

            if sum_pixels/(img.shape[0]*img.shape[1]) > 0.05 and sum_pixels/(img.shape[0]*img.shape[1]) < 0.3 and \
                sum_pixels/((mask_bbox[2]-mask_bbox[0])*(mask_bbox[3]-mask_bbox[1])) > 0.7: #  and 
                #cv2.imshow("org", original_mask)
                tot_amount = sum_pixels/(img.shape[0]*img.shape[1])
                if tot_amount < best_and_lowest:
                    best_and_lowest = tot_amount
                    best_mask = [mask, bbox]
                #cv2.imshow("corrected,", original_mask)
                #cv2.imshow("mask", mask)
                #cv2.waitKey(0)
                #return original_mask
    
    if type(best_mask) == list:
        mask, bbox = best_mask
        original_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = (mask[1:-1,1:-1]==0).astype(np.uint8)*255
        original_mask = cv2.morphologyEx(original_mask, cv2.MORPH_OPEN, np.ones((img.shape[0]//10,15)))
    
    return original_mask

def pbm_segmentation(img, margin=0.01, threshold=0.000001):
    """ Probability-based segmentation. Model the background with a multivariate gaussian distribution. 
            img: Image to be segmented
            margin: Part of image being certainly part of the background 
            threshold: Threshold to reject a pixel as not being part of the background
        returns: Mask image, indicating the painting part in the image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) 
    # Mask based on a bivariate gaussian distribution
    mask = compute_mask_gaussian_HSL(img, margin, threshold)
    
    # Compute mask based on connected components
    results = mask_segmentation_cc(img, mask)
    return results

def mask_segmentation_cc(img, mask):
    """ Take two biggest connected components from mask, making sure that they do not overlap"""
    kernel = np.ones((img.shape[0]//55, img.shape[1]//55), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, borderValue=0)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[:, -1]

    top_two_conn_comp_idx = sizes.argsort()
    top_two_conn_comp_idx = top_two_conn_comp_idx[top_two_conn_comp_idx!=0]
    if len(top_two_conn_comp_idx) > 1:
        top_two_conn_comp_idx = top_two_conn_comp_idx[[-2,-1]][::-1]
    else:
        top_two_conn_comp_idx = top_two_conn_comp_idx[[-1]][::-1]
    
    idxs = [idx for idx in top_two_conn_comp_idx]

    bc = np.zeros(output.shape)
    bc[output == idxs[0]] = 255
    bc = create_convex_painting(mask, bc)

    if len(bc[bc!=0]) == 0:
        return [np.zeros_like(mask), [[0,0,img.shape[0],img.shape[1]]], [np.ones_like(mask)]]

    if len(idxs) > 1:
        sbc = np.zeros(output.shape)
        sbc[output == idxs[1]] = 255
        sbc = create_convex_painting(mask, sbc)

    bboxes = [get_bbox(bc)]
    resulting_masks = bc
    splitted_resulting_masks = [bc]

    # Second painting if first one does not take most part + more or less a rectangular shape + no IoU
    if len(idxs) > 1:
        if not takes_most_part_image(bc) and regular_shape(sbc) and check_no_iou(bc, sbc):
            bboxes.append(get_bbox(sbc))
            resulting_masks = np.logical_or(resulting_masks==255, sbc==255).astype(np.uint8)*255
            splitted_resulting_masks.append(sbc)
    
    return resulting_masks, bboxes, splitted_resulting_masks

def create_convex_painting(mask, component_mask):
    kernel = np.ones((5, 5), np.uint8)
    
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, borderValue=0)
    contours, hierarchy = cv2.findContours((component_mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask).astype(np.uint8)
    polished_mask = cv2.fillPoly(mask, contours, 255).astype(np.uint8)
    a = polished_mask.copy()
    
    p = int(max(mask.shape[0]/8, mask.shape[1]/8))
    polished_mask = cv2.copyMakeBorder(src=polished_mask, top=p, bottom=p, left=p, right=p, borderType=cv2.BORDER_CONSTANT, value=0) 
    size1, size2 = int(mask.shape[0]*1/32),int(mask.shape[1]*1/32)
    kernel = np.ones((size1, size2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask, cv2.MORPH_CLOSE, kernel, borderValue=0)
    size1, size2 = int(mask.shape[0]/8), int(mask.shape[1]/8)
    kernel = np.ones((size1, size2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask, cv2.MORPH_OPEN, kernel, borderValue=0)

    if len(polished_mask[polished_mask!=0]) != 0:
        rect_portion = 0.6
        x0,y0,x1,y1 = get_bbox(polished_mask)
        kernel = np.ones((int((x1-x0)*rect_portion), int((y1-y0)*rect_portion)), np.uint8)
        polished_mask = cv2.morphologyEx(polished_mask, cv2.MORPH_OPEN, kernel, borderValue=0)
    return polished_mask[p:polished_mask.shape[0]-p, p:polished_mask.shape[1]-p]

def takes_most_part_image(img):
    h_quarter, w_quarter = img.shape[0]//4, img.shape[1]//4
    return img[h_quarter, w_quarter*2] == 1 and img[h_quarter*2, w_quarter] == 1 and img[h_quarter*3, w_quarter*2] == 1 and img[h_quarter*2, w_quarter*3] == 1

def get_bbox(mask):
    num_pixel_estimation = 20
    positions = np.where(mask==255)
    hs, ws = sorted(positions[0]), sorted(positions[1])
    h_min, h_max = int(np.array(hs[:num_pixel_estimation]).mean()), int(np.array(hs[-num_pixel_estimation:]).mean())
    w_min, w_max = int(np.array(ws[:num_pixel_estimation]).mean()), int(np.array(ws[-num_pixel_estimation:]).mean())
    return [h_min, w_min, h_max, w_max]

def regular_shape(mask, threshold=0.7):
    if mask.sum() == 0:
        return False
    h_min, w_min, h_max, w_max = get_bbox(mask)
    sum_pixels = (mask[h_min:h_max, w_min:w_max]==255).astype(np.uint8).sum()
    return sum_pixels/((h_max-h_min)*(w_max-w_min)) > threshold

def check_no_iou(mask1, mask2):
    bbox1, bbox2 = get_bbox(mask1), get_bbox(mask2)
    return bb_intersection_over_union(bbox1, bbox2) < 1e-6

def compute_mask_gaussian_HSL(img, margin, threshold=0.00001):
    h_m, w_m = int(img.shape[0]*margin), int(img.shape[1]*margin)

    # Compute mean and standard deviation for each channel separately
    l_mean = (np.concatenate([img[:h_m, :, 0].reshape(-1), img[:, :w_m, 0].reshape(-1), img[img.shape[0]-h_m:, :, 0].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 0].reshape(-1)])).mean()
    a_mean = (np.concatenate([img[:h_m, :, 1].reshape(-1), img[:, :w_m, 1].reshape(-1), img[img.shape[0]-h_m:, :, 1].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 1].reshape(-1)])).mean()
    b_mean = (np.concatenate([img[:h_m, :, 2].reshape(-1), img[:, :w_m, 2].reshape(-1), img[img.shape[0]-h_m:, :, 2].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 2].reshape(-1)])).mean()

    l_std = (np.concatenate([img[:h_m, :, 0].reshape(-1), img[:, :w_m, 0].reshape(-1), img[img.shape[0]-h_m:, :, 0].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 0].reshape(-1)])).std()
    a_std = (np.concatenate([img[:h_m, :, 1].reshape(-1), img[:, :w_m, 1].reshape(-1), img[img.shape[0]-h_m:, :, 1].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 1].reshape(-1)])).std()
    b_std = (np.concatenate([img[:h_m, :, 2].reshape(-1), img[:, :w_m, 2].reshape(-1), img[img.shape[0]-h_m:, :, 2].reshape(-1), \
                            img[:, img.shape[1]-w_m:, 2].reshape(-1)])).std()

    # Model background and discard unlikely pixels
    mask = stats.norm.pdf(img[:,:,0], l_mean, l_std)*stats.norm.pdf(img[:,:,1], a_mean, a_std)*stats.norm.pdf(img[:,:,2], b_mean, b_std) < threshold

    return mask
    

def create_rectangular_painting(mask, component_mask):
    kernel = np.ones((5, 5), np.uint8)
    
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, borderValue=0)
    contours, hierarchy = cv2.findContours((component_mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask).astype(np.uint8)
    x,y,w,h = cv2.boundingRect(contours[0])
    polished_mask = np.zeros_like(mask)
    polished_mask[y:y+h, x:x+w] = 255

    p = int(max(mask.shape[0]/8, mask.shape[1]/8))
    polished_mask = cv2.copyMakeBorder(src=polished_mask, top=p, bottom=p, left=p, right=p, borderType=cv2.BORDER_CONSTANT, value=0) 
    size1, size2 = int(mask.shape[0]*1/32),int(mask.shape[1]*1/32)
    kernel = np.ones((size1, size2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask, cv2.MORPH_CLOSE, kernel, borderValue=0)
    size1, size2 = int(mask.shape[0]/8), int(mask.shape[1]/8)
    kernel = np.ones((size1, size2), np.uint8)
    polished_mask = cv2.morphologyEx(polished_mask, cv2.MORPH_OPEN, kernel, borderValue=0)

    if len(polished_mask[polished_mask!=0]) != 0:
        rect_portion = 0.6
        x0,y0,x1,y1 = get_bbox(polished_mask)
        kernel = np.ones((int((x1-x0)*rect_portion), int((y1-y0)*rect_portion)), np.uint8)
        polished_mask = cv2.morphologyEx(polished_mask, cv2.MORPH_OPENss, kernel, borderValue=0)
    return polished_mask[p:polished_mask.shape[0]-p, p:polished_mask.shape[1]-p]

def removal_bg_text(qs_imgs, p_bg_masks, p_bg_annotations, p_text_annotations):
    resulting_images = []
    for i in range(len(p_bg_masks)):
        painting_imgs = []
        for j in range(len(p_bg_masks[i])):
            bbox_painting = p_bg_annotations[i][j]
            cropped_img = qs_imgs[i][bbox_painting[0]:bbox_painting[2], bbox_painting[1]:bbox_painting[3]]
            bbox_text = p_text_annotations[i][j]
            bbox_text = [bbox_text[1]-bbox_painting[0], bbox_text[0]-bbox_painting[1], bbox_text[3]-bbox_painting[0], bbox_text[2]-bbox_painting[1]]
            mask = np.zeros((cropped_img.shape[0], cropped_img.shape[1])).astype(np.uint8)
            mask[bbox_text[0]:bbox_text[2], bbox_text[1]:bbox_text[3]] = 255
            cropped_img = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)
            painting_imgs.append(cropped_img)
        resulting_images.append(painting_imgs)
    return resulting_images

def removal_text(qs_imgs, p_text_annotations):
    resulting_images = []
    for i in range(len(p_text_annotations)):
        bbox_text = p_text_annotations[i][0]
        bbox_text = [bbox_text[1], bbox_text[0], bbox_text[3], bbox_text[2]]
        mask = np.zeros((qs_imgs[i].shape[0], qs_imgs[i].shape[1])).astype(np.uint8)
        mask[bbox_text[0]:bbox_text[2],bbox_text[1]:bbox_text[3]] = 255
        cropped_img = qs_imgs[i]
        cropped_img = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)
        resulting_images.append([cropped_img])

    return resulting_images

def crop_region(text_masks, mask_bboxes):
    cropped_text_masks = []
    for p_masks, box_masks in zip(text_masks, mask_bboxes):
        for p_mask, box_mask in zip(p_masks, box_masks):
            box_mask = box_mask[p_mask[0]:p_mask[2], p_mask[1]:p_mask[3]]
            cropped_text_masks.append([box_mask])

    return cropped_text_masks

PBM = "PBM"
ES = "ES"
METHODS = [PBM, ES]

MAP_METHOD = {
    METHODS[0]: pbm_segmentation,
    METHODS[1]: edge_segmentation
}

def get_method(method=1):
    return MAP_METHOD[method]