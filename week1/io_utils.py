import pickle
import os
import glob
import cv2
import numpy as np
from time import time

QUERY_SET_ANN_PATH = "gt_corresps.pkl"
DB_FOLDER = "BBDD" # 2GB of RAM to load
QS_FOLDER = "qsd1_w1"

def load_images(db_path, ext="jpg"):
    """ Load images from path """
    file_list = glob.glob(os.path.join(db_path, "*."+ext))
    file_list.sort(key= lambda x: int(x.split(".")[-2][-5:]))
    img_list = [cv2.imread(img_p)[...,::-1] for img_p in file_list]
    return img_list

def load_db(db_path):
    """ Load DB images """
    img_list = load_images(db_path)
    labels = list(range(len(img_list)))
    return img_list, labels

def load_annotations(anno_path):
    """ Load annotations from path for query set. List of groundtruths [7, 2, 3, ..., 10] """
    if os.path.exists(anno_path):
        fd = open(anno_path, "rb")
        annotations = pickle.load(fd)
        return np.array(annotations).reshape(-1)
    else:
        return None

def mask_imgs_to_single_channel(img_list):
    return [(img[:,:,0] != 0).astype(np.uint8) for img in img_list]

def load_query_set(db_path):
    """ Load query set db and annotations """ 
    img_list = load_images(db_path)
    mask_list = mask_imgs_to_single_channel(load_images(db_path, "png"))
    labels = load_annotations(os.path.join(db_path, QUERY_SET_ANN_PATH))
    return img_list, labels, mask_list

def to_pkl(results, result_path):
    """ Write results to pkl file in result_path """
    pass