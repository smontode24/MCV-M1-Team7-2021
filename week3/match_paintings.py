import argparse
import sys
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from text_segmentation import *
from metrics import *
from evaluation.mask_evaluation import *
from evaluation.retrieval_evaluation import *
from text_recognition import extract_text_from_imgs
from time import time
from io_utils import *
from debug_utils import *
from filtering import *
import numpy as np
import cv2
import os

def parse_input_args():
    """ Parse program arguments """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path", type=str,
                        help="path to folder with db + query set")
    parser.add_argument("db_path", type=str,
                        help="db folder name")
    parser.add_argument("qs_path", type=str,
                        help="query set folder name")
    parser.add_argument("--masking", default=1, type=int,
                        help="apply masking to paintings to remove background (0 no, 1 yes)")
    parser.add_argument("--text_removal", default=1, type=int,
                        help="apply masking to paintings to remove background (0 no, 1 yes)")
    parser.add_argument("--output_pkl", type=str,
                        help="output file to write pkl file with results")
    parser.add_argument("--opkl_text", type=str,
                        help="output file to write pkl file with text detections")
    parser.add_argument("--output_mask", type=str,
                        help="output folder to save predicted masks")
    #parser.add_argument("--matching_measure", type=str, default="l1_dist",
    #                    help="matching measures [l1_dist, l2_dist, js_div, hellinger, levensthein]")
    parser.add_argument("--matching_measures", type=str, nargs='+', default=["l1_dist"], 
                        help="matching measures [l1_dist, l2_dist, js_div, hellinger, levensthein], it must correspond to the measure used in each \
                              retrieval method.")
    #parser.add_argument("-rm", "--retrieval_method", default="CBHC",
    #                    help="which method to use for painting retrieval")
    parser.add_argument("-mm", "--masking_method", default="ES",
                        help="which method to use for painting retrieval [ES, PBM]")
    parser.add_argument("-tm", "--text_method", default="MM",
                        help="which method to use for text masking")
    parser.add_argument("-ft", "--filter_type", default="median",
                        help="denoising technique")
    parser.add_argument("-rmm", type=str, nargs="+", default=["text", "HC"], # Will substitute rm
                        help="List of methods to use for comparison [OCR:Author name, CH:Color histogram, HOG, LBP, ...]")
    parser.add_argument("-w", "--weights", nargs="+", type=int, default=1, # Will substitute rm
                        help="Weight of each method to use")
    parser.add_argument("-d", "--debug", default=0, type=int,
                       help="shows images and some steps for debugging (0 no, 1 yes)")
    parser.add_argument("--use_boxes_annotations", default=0, type=int,
                       help="use boxes annotations (0 no, 1 yes)")

    args = parser.parse_args()

    #Awesome checks to avoid "STUPID" combinations of arguments
    # checks for "qsd1"
    if "qsd1" in args.qs_path:
        # not usage of MASKING
        # this check avoids => ValueError: cannot convert float NaN to integer
        if args.masking:
            print('Generally speaking, files with "qsd1" are unmasked')
            print('Try --masking 0')
            sys.exit("ABORTING ##1")

    if args.debug == 1:
        setDebugMode(True)
    return args

def match_paintings(args):

    if isDebug():
        t0 = time()
    
    # Load DB
    db_imgs, db_annotations, db_authors = load_db(path.join(args.ds_path, args.db_path))
    qs_imgs, qs_gts, qs_mask_list, qs_text_bboxes = load_query_set(path.join(args.ds_path, args.qs_path))


    # Previous checks: if db_imgs = 0
    if len(db_imgs)==0:
        print('MAIN DB: Nothing was found on path: ', args.db_path)
        print('Check that is correct')
        sys.exit("ABORTING: Nothing_in_DB_path")
    #TODO
    ''' 
    if len(db_annotations)==0:
        print('ANNOTATIONS: Nothing was found on path: ', args.db_anotations)
        print('Check that is correct')
        # sys.exit("ABORTING: Nothing_in_ANNOTATIONS_path")

    if len(db_authors) == 0:
        print('AUTHORS: Nothing was found on path: ', args.db_authors)
        print('Check that is correct')
        # sys.exit("ABORTING: Nothing_in_AUTHORS_path")
    '''
    # Previous checks: if qs_imgs = 0
    if len(qs_imgs) == 0:
        print('QDS: Nothing was found on path: ', args.qs_path)
        print('Check that is correct')
        sys.exit("ABORTING: Nothing_in_QDS_path")


    # Remove noise (e.g., use median filter k=7)
    #qs_imgs = denoise_images(qs_imgs, "median")

    if isDebug():
        #print("Time to load DB and query set:", time()-t0, "s")
        #print("Size of qs_imgs:         ", len(qs_imgs))
        #print("Size of qs_gts :         ", len(qs_gts))
        #print("Size of qs_mask_list:    ", len(qs_mask_list))
        #print("Size of qs_text_bboxes:  ", len(qs_text_bboxes))
        t0 = time()
    denoise_images(qs_imgs, args.filter_type)
    # Obtain painting region from images
    if args.masking:
        # Obtain masks for the paintings. -> Denoise on copy only because it affects text recognition a lot
        mask_file = path.join(args.ds_path, args.qs_path, "mask_predictions.pkl")
        if not os.path.isfile(mask_file):
            masked_regions, mask_bboxes, separated_bg_masks = bg_mask(denoise_images(qs_imgs, args.filter_type), args.masking_method)
            to_pkl([masked_regions, mask_bboxes, separated_bg_masks], path.join(args.ds_path, args.qs_path, "mask_predictions.pkl"))
        else:
            masked_regions, mask_bboxes, separated_bg_masks = load_plain_pkl(mask_file)
            print("Loading previous mask!")
    else:
        print("DEBUGGING")
        # Convert list of images into list of list of images (as without masking there will be a single painting,                                                                                                                                                                                                                                                                           
        # we just have to add a list structure with one image)
        masked_regions, mask_bboxes, separated_bg_masks = [[[np.ones((image.shape[0], image.shape[1])).astype(bool)] for image in qs_imgs], \
                          [[[0, 0, image.shape[0], image.shape[1]]] for image in qs_imgs], \
                          [[np.ones((image.shape[0], image.shape[1])).astype(bool)] for image in qs_imgs]]

    if isDebug():
        print("Extracted masks in:", time()-t0,"s")

    if args.text_removal:
        # Crop paintings rectangularly to later extract text
        cropped_qs_imgs = crop_painting_for_text(qs_imgs, mask_bboxes)

        if args.use_boxes_annotations == 0:
            # Compute for each painting its text segmentation
            text_masks, text_regions, relative_boxes = estimate_text_mask(cropped_qs_imgs, mask_bboxes, args.text_method, qs_imgs)
        else:
            text_masks, text_regions, relative_boxes = process_gt_text_mask(qs_text_bboxes, mask_bboxes, cropped_qs_imgs)

        # Text extractor
        painting_text = extract_text_from_imgs(cropped_qs_imgs, relative_boxes)

    # Perform painting matching
    t0 = time()

    # Denoise images before extracting descriptors
    qs_imgs = denoise_images(qs_imgs, args.filter_type)

    # Clear bg and text
    if args.text_removal and args.masking:

        if type(qs_gts) == list: # Reorder both predictions and grountruths: left-right and top-bottom
            bg_reord = sort_annotations_and_predictions(qs_gts, qs_text_bboxes, text_regions, masked_regions=separated_bg_masks, masked_boxes=mask_bboxes, text_mask=text_masks)
            qs_gts, qs_text_bboxes, text_regions, separated_bg_masks, mask_bboxes, text_masks = bg_reord
        else: # Reorder only predictions: left-right and top-bottom
            text_regions, separated_bg_masks, mask_bboxes, text_masks = sort_predictions_no_gt(text_regions, masked_regions=separated_bg_masks, masked_boxes=mask_bboxes, text_mask=text_masks)

        # Remove background and text
        qs_imgs_refined = removal_bg_text(qs_imgs, separated_bg_masks, mask_bboxes, text_regions, "CBHC") # Temporal CBHC 
    else:
        # Remove only text
        qs_imgs_refined = removal_text(qs_imgs, text_regions, "CBHC")

    # Generate query set assignments
    metrics_f = [get_measure(m) for m in args.matching_measures]
    assignments = painting_matching_ml(qs_imgs_refined, db_imgs, args.rmm, text_masks, painting_text, db_authors, metrics_f, weights=args.weights)
    print("Matching in", time()-t0,"s")

    # If query set annotations are available, evaluate
    # Evalute mIoU
    if qs_text_bboxes != None:
        mIoU = text_mIoU(text_regions, qs_text_bboxes) # Predicted bboxes / groundtruth bboxes
        if np.isnan(mIoU):  # Check if do I have some results or not
            print("Hey, I shouldn't be here: what is the sign used for empty texts??")
        else:               # show the text found
            print("Mean IoU text mask:", mIoU)

    # Evaluate painting mask
    if len(qs_mask_list) > 0 and args.masking:
        #if isDebug():
        #    for i in range(30):
        #        cv2.imshow("img", qs_imgs[i])
        #        cv2.imshow("mask_pred", masked_regions[0][i])
        #        cv2.imshow("mask_anno", qs_mask_list[i])
        #        cv2.imshow("diff", masked_regions[0][i]-qs_mask_list[i])
        #        cv2.waitKey(0)
        
        pre, rec, acc, f1 = mask_metrics(masked_regions, qs_mask_list)

        print("Precision", pre)
        print("Recall", rec)
        print("Accuracy", acc)
        print("F1-measure", f1)

    # Evaluate painting matching
    if type(qs_gts) == list:
        qs_gts = reformat_qs_gts(qs_gts)
        qs_gts = np.array(qs_gts).reshape(-1, 1)        
        if qs_gts[0].dtype == 'O':
            qs_gts = np.concatenate([[q for q in qs_gt[0]] for qs_gt in qs_gts]).reshape(-1, 1)
        map_at_1 = mapk(qs_gts, assignments, k=1)
        map_at_5 = mapk(qs_gts, assignments, k=5)

        print("MAP@1:", map_at_1)
        print("MAP@5:", map_at_5)                                                                                                                                                                                                                                                                                               
   
    print("Done")

    # Save to pkl file if necessary
    if args.output_pkl:
        # Save assignments
        assignments = reformat_assignments_to_save(assignments, text_regions)
        to_pkl(assignments, args.output_pkl)

    # Save text bounding boxes
    if args.opkl_text:
        to_pkl(text_regions, args.opkl_text)

    # Save mask images if necessary
    if args.output_mask:
        save_masks(masked_regions, args.output_mask)

if __name__ == "__main__":
    parsed_args = parse_input_args()
    match_paintings(parsed_args)

