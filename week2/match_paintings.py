import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from text_segmentation import *
from metrics import *
from evaluation.mask_evaluation import *
from evaluation.retrieval_evaluation import *
from time import time
from io_utils import *
from debug_utils import *
import numpy as np
import cv2

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
    parser.add_argument("--matching_measure", type=str, default="l1_dist",
                        help="matching measures [l1_dist, l2_dist, js_div, hellinger]")
    parser.add_argument("-rm", "--retrieval_method", default="CBHC",
                        help="which method to use for painting retrieval")
    parser.add_argument("-mm", "--masking_method", default="PBM",
                        help="which method to use for painting retrieval")
    parser.add_argument("-tm", "--text_method", default="SM",
                        help="which method to use for text masking")
    parser.add_argument("-d", "--debug", default=0, type=int,
                       help="shows images and some steps for debugging (0 no, 1 yes)")

    args = parser.parse_args()

    if args.debug == 1:
        setDebugMode(True)
    return args

def match_paintings(args):
    
    if isDebug():
        t0 = time()
    
    # Load DB
    db_imgs, db_annotations = load_db(path.join(args.ds_path, args.db_path))
    qs_imgs, qs_gts, qs_mask_list, qs_text_bboxes = load_query_set(path.join(args.ds_path, args.qs_path))

    if isDebug():
        #print("Time to load DB and query set:", time()-t0, "s")
        #print("Size of qs_imgs:         ", len(qs_imgs))
        #print("Size of qs_gts :         ", len(qs_gts))
        #print("Size of qs_mask_list:    ", len(qs_mask_list))
        #print("Size of qs_text_bboxes:  ", len(qs_text_bboxes))
        t0 = time()

    # Obtain painting region from images
    if args.masking:
        # Obtain masks for the paintings
        masked_regions = bg_mask(qs_imgs, args.masking_method)
    else:
        print("DEBUGGING")
        # Convert list of images into list of list of images (as without masking there will be a single painting, 
        # we just have to add a list structure with one image)
        masked_regions = [[[np.ones((image.shape[0], image.shape[1])).astype(bool)] for image in qs_imgs], \
                          [[[0, 0, image.shape[0], image.shape[1]]] for image in qs_imgs], \
                          [[np.ones((image.shape[0], image.shape[1])).astype(bool)] for image in qs_imgs]]

    if isDebug():
        print("Extracted masks in:", time()-t0,"s")

    if args.text_removal:
        # Crop paintings rectangularly to later extract text
        cropped_qs_imgs = crop_painting_for_text(qs_imgs, masked_regions[1])

        # Compute for each painting its text segmentation
        text_regions = estimate_text_mask(cropped_qs_imgs, masked_regions[1], args.text_method, qs_imgs)

    # Perform painting matching
    t0 = time()

    # Clear bg and text
    if args.text_removal and args.masking:
        if type(qs_gts) == list:
            bg_reord = sort_annotations_and_predictions(qs_gts, qs_text_bboxes, text_regions[1], masked_regions=masked_regions[2], masked_boxes=masked_regions[1])
            qs_gts, qs_text_bboxes, text_regions[1], masked_regions[2], masked_regions[1] = bg_reord
        else:
            bg_reord = sort_predictions_no_gt(text_regions[1], masked_regions=masked_regions[2], masked_boxes=masked_regions[1])
        qs_imgs_refined = removal_bg_text(qs_imgs, masked_regions[2], masked_regions[1], text_regions[1], args.retrieval_method)
    else:
        #qs_gts, qs_text_bboxes, text_regions[1] = sort_annotations_and_predictions(qs_gts, qs_text_bboxes, text_regions[1])
        qs_imgs_refined = removal_text(qs_imgs, text_regions[1], args.retrieval_method)

    assignments = painting_matching(qs_imgs_refined, db_imgs, args.retrieval_method, metric=get_measure(args.matching_measure))

    print("Matching in", time()-t0,"s")

    # If query set annotations are available, evaluate
    # Evalute mIoU
    if qs_text_bboxes != None:
        mIoU = text_mIoU(text_regions[1], qs_text_bboxes)
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
        
        pre, rec, acc, f1 = mask_metrics(masked_regions[0], qs_mask_list)

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
        assignments = reformat_assignments_to_save(assignments, text_regions[1])
        #assignments = assignments[:, :10].tolist()
        to_pkl(assignments, args.output_pkl)

    # Save text bounding boxes
    if args.opkl_text:
        to_pkl(text_regions[1], args.opkl_text)

    # Save mask images if necessary
    if args.output_mask:
        save_masks(masked_regions[0], args.output_mask)

if __name__ == "__main__":
    parsed_args = parse_input_args()
    match_paintings(parsed_args)

