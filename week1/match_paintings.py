import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *

def parse_input_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path",
                        help="path to dataset")
    parser.add_argument("db_path",
                        help="path to dataset")
    parser.add_argument("qs_path",
                        help="path to dataset")
    parser.add_argument("--output_pkl",
                        help="output of pkl file with results")
    parser.add_argument("--output_mask",
                        help="output of mask folder to write results")
    parser.add_argument("-m", "--method", default=1,
                        help="which method to use")
    
    args = parser.parse_args()
    return args

def match_paintings(args):
    # Load DB
    db_imgs, db_annotations = load_db(path.join(args.ds_path, args.db_path))
    qs_imgs, qs_gts, qs_mask_list = load_query_set(path.join(args.ds_path, args.qs_path))

    # Obtain painting region from images
    masked_regions = bg_mask(db_imgs) # TODO
    # Perform painting matching
    assignments = painting_matching(qs_imgs, db_imgs) # TODO

    # If query set annotations are available, evaluate 
    # TODO: Implement in evaluation

    # TODO: Save to pkl file if necessary
    if args.output_pkl:
        pass

    # TODO: Save mask images if necessary
    if args.output_mask:
        pass

if __name__ == "__main__":
    parsed_args = parse_input_args()
    match_paintings(parsed_args)