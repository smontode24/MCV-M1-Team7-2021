import numpy as np
import sys
from metrics import *
import cv2
from debug_utils import *
from tqdm import tqdm
from inspect import signature
from descriptors import *

def painting_matching(imgs, db_imgs, method_name, metric=js_div, splits=30, max_rank=10): 
    """ Obtain query images matches.
        Params:
            - imgs: query set of images [img1, img2,...]
            - db_imgs: database images
            - metric: Similarity measure to quantify the distance between to histograms
            - method_name: method to apply
            - metric: l1_dist,... (which distance / similarity to use)
        Returns: 
            Top k matches for each image of the query set in the database
    """
    matching_method = get_method(method_name)
    tmp_img_format = []
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            tmp_img_format.append(imgs[i][j])

    db_img_splits = [i*len(db_imgs)//splits for i in range(splits-1)]
    
    scores = []
    query_descriptors = np.array([matching_method(img) for img in tmp_img_format])
    print("Starting db extraction + matching")
    if splits > 1:
        for split in tqdm(range(splits-2)):
            db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs[db_img_splits[split]:db_img_splits[split+1]]])
            scores.append(metric(query_descriptors, db_descriptors))
        
        db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs[db_img_splits[-1]:]])
        scores.append(metric(query_descriptors, db_descriptors))
        
        # concatenate all the results
        scores = np.concatenate(scores, 1)
    else:
        db_descriptors = np.array([matching_method(db_img) for db_img in db_imgs])
        scores = metric(query_descriptors, db_descriptors)
    
    top_k_matches = np.argpartition(scores, list(range(max_rank)))[:, :max_rank]
    return top_k_matches

def painting_matching_wmasks(imgs, db_imgs, method_name, text_masks, metric=js_div, splits=30, max_rank=10): 
    """ Obtain query images matches.
        Params:
            - imgs: query set of images [img1, img2,...]
            - db_imgs: database images
            - metric: Similarity measure to quantify the distance between to histograms
            - method_name: method to apply
            - metric: l1_dist,... (which distance / similarity to use)
        Returns: 
            Top k matches for each image of the query set in the database
    """
    matching_method = get_method(method_name)
    tmp_img_format = []
    tmp_mask_format = []
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            tmp_img_format.append(imgs[i][j])
            tmp_mask_format.append(text_masks[i][j])

    db_img_splits = [i*len(db_imgs)//splits for i in range(splits-1)]
    
    scores = []
    query_descriptors = np.array([mrhm(img, mask) for img, mask in zip(tmp_img_format, tmp_mask_format)], dtype=np.ndarray)

    print("Starting db extraction + matching")
    if splits > 1:
        for split in tqdm(range(splits-2)):
            db_descriptors = np.array([mrhm(db_img) for db_img in db_imgs[db_img_splits[split]:db_img_splits[split+1]]])
            scores.append(metric(query_descriptors, db_descriptors))
        
        db_descriptors = np.array([mrhm(db_img) for db_img in db_imgs[db_img_splits[-1]:]])
        scores.append(metric(query_descriptors, db_descriptors))
        
        # concatenate all the results
        scores = np.concatenate(scores, 1)
    else:
        db_descriptors = np.array([mrhm(db_img) for db_img in db_imgs])
        scores = metric(query_descriptors, db_descriptors)
    
    top_k_matches = np.argpartition(scores, list(range(max_rank)))[:, :max_rank]
    return top_k_matches

# WIP: Multiple criteria
def painting_matching_ml(imgs, db_imgs, method_list, text_masks, author_text, gt_text, metrics, weights, splits=30, max_rank=10): 
    """ Obtain query images matches.
        Params:
            - imgs: query set of images [img1, img2,...]
            - db_imgs: database images
            - method_list: List of methods: DCT, HOG, OCR, ...
            - text_masks: Textboxes masks
            - author_text: Predicted authors in the query set
            - gt_text: Groundtruths of authors for DB 
            - metrics: List of similarities for method comparison (e.g., if using "text" and "HC" as methods, according metrics would be: "levenshtein" and "l1_dist")  
            - weights: Multiply each method by its importance (i.e. maybe we want to give more importance to HOG than HC)
        Returns: 
            Top k matches for each image of the query set in the database
    """
    descriptor_extractors = [get_descriptor_extractor(method_name) for method_name in method_list]
    tmp_img_format = []
    tmp_mask_format = []
    tmp_text_format = []
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            tmp_img_format.append(imgs[i][j])
            tmp_mask_format.append(text_masks[i][j])
            tmp_text_format.append(author_text[i][j])

    #db_imgs = [img[0] for img in db_imgs]
    db_img_splits = [i*len(db_imgs)//splits for i in range(splits-1)]
    
    scores = []
    query_descriptors = extract_descriptors(tmp_img_format, descriptor_extractors, method_list, tmp_text_format, tmp_mask_format) 
    #np.array([extract_descriptors(img, matching_methods, mask) for img, mask in zip(tmp_img_format, tmp_mask_format)])
    print("Starting db extraction + matching")
    for split in tqdm(range(splits-2)):
        db_descriptors = extract_descriptors(db_imgs[db_img_splits[split]:db_img_splits[split+1]], descriptor_extractors, method_list, gt_text[db_img_splits[split]:db_img_splits[split+1]], None) #np.array([mrhm(db_img) for db_img in db_imgs[db_img_splits[split]:db_img_splits[split+1]]])
        scores.append(compare_descriptors(query_descriptors, db_descriptors, metrics, method_list, weights))
    # compare_descriptors(query_descriptors, db_descriptors, descriptor_comp_methods, descriptor_names, weights)
    db_descriptors = extract_descriptors(db_imgs[db_img_splits[-1]:], descriptor_extractors, method_list, gt_text[db_img_splits[-1]:], None)
    scores.append(compare_descriptors(query_descriptors, db_descriptors, metrics, method_list, weights))
    
    # concatenate all the results
    scores = np.concatenate(scores, 1)
    """ else:
        db_descriptors = np.array([mrhm(db_img) for db_img in db_imgs])
        scores = compare_descriptors(query_descriptors, db_descriptors, metrics, method_list, weights) """
    
    top_k_matches = np.argpartition(scores, list(range(max_rank)))[:, :max_rank]
    return top_k_matches

def extract_descriptors(imgs, descriptor_extractors, descriptor_names, author_text, text_masks):
    """ Extract descriptors from imgs.
        params:
            imgs: List of images
            descriptor_extractors: List of descriptor extractors
            descriptor_names: List of descriptor names
            author_text: List of painting authors 
            text_masks: List of text boxes masks
        return: List of descriptors with the descriptor of each image 
    """
    descriptor = []
    for descriptor_name, descriptor_extractor in zip(descriptor_names, descriptor_extractors):
        if descriptor_name == "text": # If text, only need to add the processed text in the descriptor
            text_descriptors = []
            for text_description in author_text:
                text_descriptors.append(text_description)
            descriptor.append(text_descriptors)
        else:
            if type(text_masks) == list:
                mask_arg = list(signature(descriptor_extractor).parameters.keys())[1] == "mask"
                if mask_arg:
                    descriptor.append(np.stack([descriptor_extractor(img, text_mask) for img, text_mask in zip(imgs, text_masks)]))
                else:
                    descriptor.append(np.stack([descriptor_extractor(img) for img in imgs]))
            else:
                descriptor.append(np.stack([descriptor_extractor(img) for img in imgs]))
    return descriptor

def compare_descriptors(query_descriptors, db_descriptors, descriptor_comp_methods, descriptor_names, weights):
    """ Extract descriptors from imgs.
        params:
            query_descriptors: List of query set descriptors (each descriptor is 30xd)
            db_descriptors: List of DB descriptors (each descriptor is 30xd)
            descriptor_extractors: List of descriptor extractors
            descriptor_comp_methods: List of measures to compare descriptors
            descriptor_names: List of descriptor names 
            weights: List of weights. This indicates the importance of each descriptor. (e.g., [1, 3.5, 0.5])
        return: List of descriptors with the descriptor of each image 
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    scores = np.zeros((len(query_descriptors[0]), len(db_descriptors[0])))
    num_descriptor = 0
    try:
        for query_descriptor, db_descriptor, descriptor_name in zip(query_descriptors, db_descriptors, descriptor_names):
            d_score = descriptor_comp_methods[num_descriptor](query_descriptor, db_descriptor)
            if descriptor_name != "text":
                d_score = d_score / (16*16)

            scores += weights[num_descriptor] * d_score
            #print(weights[num_descriptor], d_score)
            num_descriptor += 1
    except TypeError:
        print("Maybe you've missed to use some --weights in your arguments")
        print('Try at least "--weights 1" to continue the execution')
        sys.exit("ABORTING ##2")
    return scores


####################
# Matching methods #
####################
TC_name = "text"
HC_name = "HC"
HOG_name = "HOG"
DCT_name = "DCT"
LBP_name = "LBP"

OPTIONS = [TC_name, HC_name, HOG_name, DCT_name, LBP_name]

METHOD_MAPPING_EXTR = {
    OPTIONS[0]: TC,
    OPTIONS[1]: celled_2dhist,
    OPTIONS[2]: HOG,
    OPTIONS[3]: DCT,
    OPTIONS[4]: LBP
}

def get_descriptor_extractor(method):
    return METHOD_MAPPING_EXTR[method]
