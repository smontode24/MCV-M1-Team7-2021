import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from metrics import *
from evaluation.mask_evaluation import *
from evaluation.retrieval_evaluation import *
from time import time
from io_utils import *

def parse_input_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path", type=str,
                        help="path to folder with db + query set")
    parser.add_argument("db_path", type=str,
                        help="db folder name")
    parser.add_argument("qs_path", type=str,
                        help="query set folder name")
    parser.add_argument("--masking", default=1, type=int,
                        help="apply masking to paintings to remove background (0 no, 1 yes)")
    parser.add_argument("--output_pkl", type=str,
                        help="output file to write pkl file with results")
    parser.add_argument("--output_mask", type=str,
                        help="output folder to save predicted masks")
    parser.add_argument("-rm", "--retrieval_method", default="twodcelled",
                        help="which method to use for painting retrieval")
    parser.add_argument("-mm", "--masking_method", default="cbhs",
                        help="which method to use for painting retrieval")
    
    args = parser.parse_args()
    return args

def match_paintings(args):
    # Load DB
    db_imgs, db_annotations = load_db(path.join(args.ds_path, args.db_path))
    qs_imgs, qs_gts, qs_mask_list = load_query_set(path.join(args.ds_path, args.qs_path))

    # Obtain painting region from images
    if args.masking:
        # Obtain masks for the paintings
        masked_regions = bg_mask(qs_imgs, args.masking_method)
        # Apply the mask on images
        qs_imgs = apply_mask(qs_imgs, masked_regions, args.masking_method)

    # Perform painting matching
    t0 = time()
    assignments = painting_matching(qs_imgs, db_imgs, args.retrieval_method)
    print("Matching in", time()-t0,"s")

    # If query set annotations are available, evaluate 
    if len(qs_mask_list) > 0:
        pre, rec, acc, f1 = mask_metrics(masked_regions, qs_mask_list)

        print("Precision", pre)
        print("Recall", rec)
        print("Accuracy", acc)
        print("F1-measure", f1)

    if len(qs_gts) > 0:
        qs_gts = qs_gts.reshape(len(qs_gts), 1)
        map_at_1 = mapk(qs_gts, assignments, k=1)
        map_at_5 = mapk(qs_gts, assignments, k=5)

        print("MAP@1:", map_at_1)
        print("MAP@5:", map_at_5)

    # Save to pkl file if necessary
    if args.output_pkl:
        to_pkl(assignments, args.output_pkl, k=10)

    # Save mask images if necessary
    if args.output_mask:
        save_masks(masked_regions, args.output_mask)

if __name__ == "__main__":
    parsed_args = parse_input_args()
    match_paintings(parsed_args)

