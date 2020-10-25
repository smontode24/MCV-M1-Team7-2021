import numpy as np
import cv2
from metrics import *
import scipy.stats as stats
from debug_utils import *
from evaluation.mask_evaluation import bb_intersection_over_union
from tqdm import tqdm

def bg_mask(query_imgs, method): 
    """ Obtain a mask for each image in the list of images query_img. The method will be determined by the passed method argument.
    Available methods are: "CBHS" and "PBM". 
        params: 
            query_imgs: List of images of the query set
            method: ["CBHS", "PBM"]
        returns:
            List of masks [2D images with 1 channel]. A pixel = 0 means background, and pixel = 255 painting
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

def pbm_segmentation(img, margin=0.02, threshold=0.000001):
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
    
    kernel = np.ones((img.shape[0]//50, img.shape[1]//50), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, borderValue=0)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[:, -1]

    top_two_conn_comp_idx = sizes.argsort()
    top_two_conn_comp_idx = top_two_conn_comp_idx[top_two_conn_comp_idx!=0][[-2,-1]][::-1]

    idxs = [idx for idx in top_two_conn_comp_idx]

    bc = np.zeros(output.shape)
    bc[output == idxs[0]] = 255
    bc = create_convex_painting(mask, bc)

    sbc = np.zeros(output.shape)
    sbc[output == idxs[1]] = 255
    sbc = create_convex_painting(mask, sbc)

    bboxes = [get_bbox(bc)]
    resulting_masks = bc
    splitted_resulting_masks = [bc]

    # Second painting if first one does not take most part + more or less a rectangular shape + no IoU
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

def compute_mask_gaussian_HSL(img, margin, threshold=0.000001):
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

def removal_bg_text(qs_imgs, p_bg_masks, p_bg_annotations, p_text_annotations, method_matching):
    resulting_images = []
    if method_matching == "CBHC" or method_matching == "CBHCM":
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
    else:
        for i in range(len(p_bg_masks)):
            painting_imgs = []
            for j in range(len(p_bg_masks[i])):
                bbox_painting = p_bg_annotations[i][j]
                mask_bg = p_bg_masks[i][j].astype(bool)
                bbox_text = p_text_annotations[i][j]
                bbox_text = [bbox_text[1], bbox_text[0], bbox_text[3], bbox_text[2]]
                mask = np.zeros((mask_bg.shape[0], mask_bg.shape[1])).astype(bool)
                mask[bbox_text[0]:bbox_text[2], bbox_text[1]:bbox_text[3]] = True
                total_mask = np.logical_or(mask_bg, mask)
                painting_imgs.append(qs_imgs[i][total_mask])
            resulting_images.append(painting_imgs)
    return resulting_images

def removal_text(qs_imgs, p_text_annotations, method_matching):
    resulting_images = []
    if method_matching == "CBHC" or method_matching == "CBHCM":
        for i in range(len(p_text_annotations)):
            bbox_text = p_text_annotations[i][0]
            bbox_text = [bbox_text[1], bbox_text[0], bbox_text[3], bbox_text[2]]
            mask = np.zeros((qs_imgs[i].shape[0], qs_imgs[i].shape[1])).astype(np.uint8)
            mask[bbox_text[0]:bbox_text[2],bbox_text[1]:bbox_text[3]] = 255
            cropped_img = qs_imgs[i]
            cropped_img = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)
            resulting_images.append([cropped_img])
    else:
        for i in range(len(p_text_annotations)):
            bbox_text = p_text_annotations[i][0]
            bbox_text = [bbox_text[1], bbox_text[0], bbox_text[3], bbox_text[2]]
            mask = np.zeros((qs_imgs[i].shape[0], qs_imgs[i].shape[1])).astype(np.uint8)
            mask[bbox_text[0]:bbox_text[2],bbox_text[1]:bbox_text[3]] = 255
            mask = mask!=255
            cropped_img = qs_imgs[i]
            resulting_images.append([cropped_img[mask]])

    return resulting_images

PBM = "PBM"
METHODS = [PBM]

MAP_METHOD = {
    METHODS[0]: pbm_segmentation
}

def get_method(method=1):
    return MAP_METHOD[method]