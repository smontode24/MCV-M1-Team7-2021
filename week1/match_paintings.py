import argparse
from io_utils import *
from os import path
from background_mask import *
from match_methods import *
from metrics import *
from evaluation import *

def parse_input_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path",
                        help="path to folder with db + query set")
    parser.add_argument("db_path",
                        help="db folder name")
    parser.add_argument("qs_path",
                        help="query set folder name")
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
    print(qs_gts)

    # Obtain painting region from images
    masked_regions = bg_mask(qs_imgs) # TODO
    # Perform painting matching
    assignments = painting_matching(qs_imgs, db_imgs) # TODO

    # If query set annotations are available, evaluate 
    # TODO: Implement in evaluation
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