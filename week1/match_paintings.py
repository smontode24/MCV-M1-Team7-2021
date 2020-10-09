import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from metrics import *
from evaluation.mask_evaluation import *
from evaluation.retrieval_evaluation import *
from time import time

def parse_input_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path", type=str,
                        help="path to folder with db + query set")
    parser.add_argument("db_path", type=str,
                        help="db folder name")
    parser.add_argument("qs_path", type=str,
                        help="query set folder name")
    parser.add_argument("--load_mask", default=1, type=int,
                        help="apply masking to paintings to remove background (0 no, 1 yes)")
    parser.add_argument("--masking", default=1, type=int,
                        help="apply masking to paintings to remove background (0 no, 1 yes)")
    parser.add_argument("--output_pkl", type=str,
                        help="output file to write pkl file with results")
    parser.add_argument("--output_mask", type=str,
                        help="output folder to save predicted masks")
    parser.add_argument("-m", "--method", default=1,
                        help="which method to use")
    
    args = parser.parse_args()
    return args

def match_paintings(args):
    # Load DB
    db_imgs, db_annotations = load_db(path.join(args.ds_path, args.db_path))
    qs_imgs, qs_gts, qs_mask_list = load_query_set(path.join(args.ds_path, args.qs_path), bool(args.load_mask))

    # Obtain painting region from images
    if args.masking:
        masked_regions = bg_mask(qs_imgs) # TODO
        # TODO: Apply mask

    # Perform painting matching
    t0 = time()
    assignments = painting_matching(qs_imgs, db_imgs) # TODO
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

    # TODO: Save to pkl file if necessary
    if args.output_pkl:
        pass

    # TODO: Save mask images if necessary
    if args.output_mask:
        pass

if __name__ == "__main__":
    parsed_args = parse_input_args()
    match_paintings(parsed_args)

