import argparse
import pickle
import numpy as np

def parse_input_args():
    """ Parse program arguments """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("org_text_boxes", type=str,
                        help="file to fix")

    args = parser.parse_args()
    return args

def fix_and_save(args, annotations):
    new_annotations = []
    for annotation in annotations:
        single_anno = annotation[0]
        new_annotations.append([single_anno[0][0], single_anno[0][1], single_anno[2][0], single_anno[2][1]])

    output = open(args.org_text_boxes, 'wb')
    pickle.dump(new_annotations, output)
    output.close()

def main(args):
    fd = open(args.org_text_boxes, "rb")
    annotations = pickle.load(fd)
    try:
        x0 = annotations[0][0][0]
        fix_and_save(args, annotations)
        print("Fixed!")
    except:
        print("File is already correct")
        return 

if __name__ == "__main__":
    args = parse_input_args()
    main(args)