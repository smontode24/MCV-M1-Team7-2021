import pickle
import os
import glob
import cv2
import numpy as np
from time import time
from debug_utils import *

QUERY_SET_ANN_PATH = "gt_corresps.pkl"
TEXT_BOXES_PATH = "text_boxes.pkl"

def load_images(db_path, ext="jpg"):
    """ Load images from path """
    file_list = glob.glob(os.path.join(db_path, "*."+ext))
    file_list.sort(key= lambda x: int(x.split(".")[-2][-5:]))
    img_list = [cv2.imread(img_p)[...,::-1] for img_p in file_list]
    return img_list

def load_db(db_path):
    """ Load DB images from the specified path. 
    Returns = [list of images, list of labels] """
    img_list = load_images(db_path)
    labels = list(range(len(img_list)))
    return img_list, labels

def load_gt_annotations(anno_path):
    """ Load annotations from path for query set. List of groundtruths [7, 2, 3, ..., 10] """
    annotations = load_pkl(anno_path)
    """ if annotations != None:
        annotations = np.array(annotations).reshape(-1) """
    return annotations

def load_pkl(anno_path):
    """ Load a pickle file """
    if os.path.exists(anno_path):
        fd = open(anno_path, "rb")
        annotations = pickle.load(fd)
        tmp_anno = np.concatenate(annotations)
        if len(tmp_anno.shape) == 1:
            annotations = [[anno] for anno in annotations]
        return annotations
    else:
        return None  

def mask_imgs_to_single_channel(img_list):
    """ Convert 3 channel mask image to single channel. """
    return [(img[:,:,0] != 0).astype(np.uint8) for img in img_list]

def load_query_set(db_path):
    """ Load query set db and annotations from the db_path folder. 
        Returns= [list of images, list of label correspondences in the database, list of masks (to remove background), list of text bounding boxes] """ 
    img_list = load_images(db_path)
    masks_list = mask_imgs_to_single_channel(load_images(db_path, "png"))
    labels = load_gt_annotations(os.path.join(db_path, QUERY_SET_ANN_PATH))
    text_labels = load_pkl(os.path.join(db_path, TEXT_BOXES_PATH))
    return img_list, labels, masks_list, text_labels

def to_pkl(results, result_path): 
    """ Write results to pkl file in result_path """
    
    output = open(result_path, 'wb')
    pickle.dump(results, output)
    output.close()

def save_masks(results, result_folder):
    """ Save resulting masks (results) into the the output folder indicated by result_path """
    for img_num, result in enumerate(results):
        if len(result.shape) == 2:
            mask = np.zeros((result.shape[0], result.shape[1], 3))
            for i in range(3):
                mask[:, :, i] = result
        else:
            mask = result

        name = str(img_num).zfill(5)+".png"
        cv2.imwrite(os.path.join(result_folder, name), mask)
