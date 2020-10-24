import numpy as np

#########################################
# Code from mcv-m1-code/evaluation ######
#########################################

def performance_accumulation_pixel(pixel_candidates, pixel_annotation):
    """ 
    performance_accumulation_pixel()
    Function to compute different performance indicators 
    (True Positive, False Positive, False Negative, True Negative) 
    at the pixel level
       
    [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)
       
    Parameter name      Value
    --------------      -----
    'pixel_candidates'   Binary image marking the foreground areas
    'pixel_annotation'   Binary image containing ground truth
       
    The function returns the number of True Positive (pixelTP), False Positive (pixelFP), 
    False Negative (pixelFN) and True Negative (pixelTN) pixels in the image pixel_candidates
    """
    
    pixel_candidates = np.uint64(pixel_candidates>0)
    pixel_annotation = np.uint64(pixel_annotation>0)
    
    pixelTP = np.sum(pixel_candidates & pixel_annotation)
    pixelFP = np.sum(pixel_candidates & (pixel_annotation==0))
    pixelFN = np.sum((pixel_candidates==0) & pixel_annotation)
    pixelTN = np.sum((pixel_candidates==0) & (pixel_annotation==0))


    return [pixelTP, pixelFP, pixelFN, pixelTN]



def performance_accumulation_window(detections, annotations):
    """ 
    performance_accumulation_window()
    Function to compute different performance indicators (True Positive, 
    False Positive, False Negative) at the object level.
    
    Objects are defined by means of rectangular windows circumscribing them.
    Window format is [ struct(x,y,w,h)  struct(x,y,w,h)  ... ] in both
    detections and annotations.
    
    An object is considered to be detected correctly if detection and annotation 
    windows overlap by more of 50%
    
       function [TP,FN,FP] = PerformanceAccumulationWindow(detections, annotations)
    
       Parameter name      Value
       --------------      -----
       'detections'        List of windows marking the candidate detections
       'annotations'       List of windows with the ground truth positions of the objects
    
    The function returns the number of True Positive (TP), False Positive (FP), 
    False Negative (FN) objects
    """
    
    detections_used  = np.zeros(len(detections))
    annotations_used = np.zeros(len(annotations))
    TP = 0
    for ii in range (len(annotations)):
        for jj in range (len(detections)):
            if (detections_used[jj] == 0) & (bbox_iou(annotations[ii], detections[jj]) > 0.5):
                TP = TP+1
                detections_used[jj]  = 1
                annotations_used[ii] = 1

    FN = np.sum(annotations_used==0)
    FP = np.sum(detections_used==0)

    return [TP,FN,FP]


def performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN):
    """
    performance_evaluation_pixel()
    Function to compute different performance indicators (Precision, accuracy, 
    specificity, sensitivity) at the pixel level
    
    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = PerformanceEvaluationPixel(pixelTP, pixelFP, pixelFN, pixelTN)
    
       Parameter name      Value
       --------------      -----
       'pixelTP'           Number of True  Positive pixels
       'pixelFP'           Number of False Positive pixels
       'pixelFN'           Number of False Negative pixels
       'pixelTN'           Number of True  Negative pixels
    
    The function returns the precision, accuracy, specificity and sensitivity
    """

    pixel_precision   = 0
    pixel_accuracy    = 0
    pixel_specificity = 0
    pixel_sensitivity = 0
    if (pixelTP+pixelFP) != 0:
        pixel_precision   = float(pixelTP) / float(pixelTP+pixelFP)
    if (pixelTP+pixelFP+pixelFN+pixelTN) != 0:
        pixel_accuracy    = float(pixelTP+pixelTN) / float(pixelTP+pixelFP+pixelFN+pixelTN)
    if (pixelTN+pixelFP):
        pixel_specificity = float(pixelTN) / float(pixelTN+pixelFP)
    if (pixelTP+pixelFN) != 0:
        pixel_sensitivity = float(pixelTP) / float(pixelTP+pixelFN)

    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity]


def performance_evaluation_window(TP, FN, FP):
    """
    performance_evaluation_window()
    Function to compute different performance indicators (Precision, accuracy, 
    sensitivity/recall) at the object level
    
    [precision, sensitivity, accuracy] = PerformanceEvaluationPixel(TP, FN, FP)
    
       Parameter name      Value
       --------------      -----
       'TP'                Number of True  Positive objects
       'FN'                Number of False Negative objects
       'FP'                Number of False Positive objects
       'F1'                Harmonic mean of precision and recall

    The function returns the precision, accuracy and sensitivity
    """
    
    precision   = float(TP) / float(TP+FP); # Q: What if i do not have TN?
    sensitivity = float(TP) / float(TP+FN)
    accuracy    = float(TP) / float(TP+FN+FP);
    f1 = 2*precision*sensitivity/(precision+sensitivity)

    return [precision, sensitivity, accuracy, f1]

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou

def mask_metrics(mask_predictions, mask_gts):
    """ Compute precision, recall, accuracy and F1 metric. 
        params:
            mask_predictions: Mask predictions (Gray image. 0->Background, 255->Painting)
            mask_gts: Mask groundtruths (Gray image. 0->Background, 255->Painting)
        returns:
            [precision, recall, accuracy, f1-measure]
    """

    results = []
    for mask_pred, mask_gt in zip(mask_predictions, mask_gts):
        results.append(performance_accumulation_pixel(mask_pred, mask_gt))

    results = np.array(results)
    TP, FP, FN, _ = results.sum(axis=0)
    pre, rec, acc, f1 = performance_evaluation_window(TP, FN, FP)
    return (pre, rec, acc, f1)

def text_mIoU(predictions, gts):
    return mIoU(predictions, gts)

def sort_annotations_and_predictions(qs_gts_matching, qs_gts_bboxes, pred_bboxes, masked_regions=None, masked_boxes=None):
    """ Sort annotations and predictions by bounding boxes close to the left-top corner """
    new_qs_gts_matching = []
    new_qs_gts_bboxes = []
    new_pred_bboxes = []

    if masked_regions != None:
        new_masked_regions = []
    if masked_boxes != None:
        new_masked_boxes = []
    
    for i in range(len(qs_gts_matching)):
        num_paintings = len(qs_gts_matching[i][0])
        if num_paintings > 1 and len(pred_bboxes[i]) > 1:
            # First sort real
            b1_c = box1_closer(qs_gts_bboxes[i][0], qs_gts_bboxes[i][1])
            if not b1_c:
                new_qs_gts_matching.append([qs_gts_matching[i][0][1], qs_gts_matching[i][0][0]])
                new_qs_gts_bboxes.append([qs_gts_bboxes[i][1], qs_gts_bboxes[i][0]])
            else:
                new_qs_gts_matching.append(qs_gts_matching[i])
                new_qs_gts_bboxes.append(qs_gts_bboxes[i])

            b1_c = box1_closer(pred_bboxes[i][0], pred_bboxes[i][1])
            if not b1_c:
                if masked_regions != None:
                    new_masked_regions.append([masked_regions[i][1], masked_regions[i][0]])
                    new_masked_boxes.append([masked_boxes[i][1], masked_boxes[i][0]])
                new_pred_bboxes.append([pred_bboxes[i][1], pred_bboxes[i][0]])
            else:
                if masked_regions != None:
                    new_masked_regions.append(masked_regions[i])
                    new_masked_boxes.append(masked_boxes[i])
                new_pred_bboxes.append(pred_bboxes[i])
            
        else:
            new_qs_gts_matching.append(qs_gts_matching[i])
            if masked_regions != None:
                new_masked_regions.append(masked_regions[i])
                new_masked_boxes.append(masked_boxes[i])
            new_qs_gts_bboxes.append(qs_gts_bboxes[i])
            new_pred_bboxes.append(pred_bboxes[i])

    result = [new_qs_gts_matching, new_qs_gts_bboxes, new_pred_bboxes]
    if masked_regions != None:
        result.append(new_masked_regions)
        result.append(new_masked_boxes)

    return result

def sort_predictions_no_gt(pred_bboxes, masked_regions=None, masked_boxes=None):
    """ Sort annotations and predictions by bounding boxes close to the left-top corner """
    new_pred_bboxes = []

    if masked_regions != None:
        new_masked_regions = []
    if masked_boxes != None:
        new_masked_boxes = []
    
    for i in range(len(pred_bboxes)):
        if len(pred_bboxes[i]) > 1:
            # First sort real
            b1_c = box1_closer(pred_bboxes[i][0], pred_bboxes[i][1])
            if not b1_c:
                if masked_regions != None:
                    new_masked_regions.append([masked_regions[i][1], masked_regions[i][0]])
                    new_masked_boxes.append([masked_boxes[i][1], masked_boxes[i][0]])
                new_pred_bboxes.append([pred_bboxes[i][1], pred_bboxes[i][0]])
            else:
                if masked_regions != None:
                    new_masked_regions.append(masked_regions[i])
                    new_masked_boxes.append(masked_boxes[i])
                new_pred_bboxes.append(pred_bboxes[i])
            
        else:
            if masked_regions != None:
                new_masked_regions.append(masked_regions[i])
                new_masked_boxes.append(masked_boxes[i])
            
    result = [new_pred_bboxes]
    if masked_regions != None:
        result.append(new_masked_regions)
        result.append(new_masked_boxes)

    return result

def box1_closer(box1, box2):
    return (box1[0]**2+box1[1]**2) < (box2[0]**2+box2[1]**2)
    
def reformat_qs_gts(qs_gts):
    tmp = []
    for i in range(len(qs_gts)):
        for j in range(len(qs_gts[i])):
            tmp.append(qs_gts[i][j])

    return tmp

def reformat_assignments_to_save(assignments, text_detections):
    
    k = 0
    assignments_ref = []
    for i in range(len(text_detections)):
        assignments_paintings = []
        for j in range(len(text_detections[i])):
            assignments_paintings.append(text_detections)
        assignments_ref.append(assignments_paintings)
    return assignments_ref

def mIoU(predictions, gts):
    """ Predictions """
    try:
        predictions = np.concatenate(predictions)
        gts = np.concatenate(gts)
        output = np.array([bb_intersection_over_union(prediction, gt) for prediction, gt in zip(predictions, gts)]).mean()
    except RuntimeWarning:
        print("If I have nothing, it would be hard to make the mean....")
    return output

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	x0, x1 = max(boxA[0], boxB[0]), min(boxA[2], boxB[2])
	y0, y1 = max(boxA[1], boxB[1]), min(boxA[3], boxB[3])
	
    # compute the area of intersection rectangle
	interArea = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou