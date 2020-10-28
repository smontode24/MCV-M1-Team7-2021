import numpy as np
from metrics import *
import cv2
from debug_utils import *
from tqdm import tqdm
from inspect import signature

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
    query_descriptors = np.array([mrhm(img, mask) for img, mask in zip(tmp_img_format, tmp_mask_format)])
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
    if splits > 1:
        for split in tqdm(range(splits-2)):
            db_descriptors = extract_descriptors(db_imgs[db_img_splits[split]:db_img_splits[split+1]], descriptor_extractors, method_list, gt_text[db_img_splits[split]:db_img_splits[split+1]], None) #np.array([mrhm(db_img) for db_img in db_imgs[db_img_splits[split]:db_img_splits[split+1]]])
            scores.append(compare_descriptors(query_descriptors, db_descriptors, metrics, method_list, weights))
        # compare_descriptors(query_descriptors, db_descriptors, descriptor_comp_methods, descriptor_names, weights)
        db_descriptors = extract_descriptors(db_imgs[db_img_splits[-1]:], descriptor_extractors, method_list, gt_text[db_img_splits[-1]:], None)
        scores.append(compare_descriptors(query_descriptors, db_descriptors, metrics, method_list, weights))
        
        # concatenate all the results
        scores = np.concatenate(scores, 1)
    else:
        db_descriptors = np.array([mrhm(db_img) for db_img in db_imgs])
        scores = compare_descriptors(query_descriptors, db_descriptors, metrics, method_list, weights)
    
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
        elif descriptor_name == "HC":
            if type(text_masks) == list:
                mask_arg = list(signature(descriptor_extractor).parameters.keys())[1] == "mask"
                if mask_arg:
                    descriptor.append(np.stack([descriptor_extractor(img, text_mask) for img, text_mask in zip(imgs, text_masks)]))
                else:
                    descriptor.append(np.stack([descriptor_extractor(img) for img in imgs]))
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
    for query_descriptor, db_descriptor, descriptor_name in zip(query_descriptors, db_descriptors, descriptor_names):
        scores += weights[num_descriptor] * descriptor_comp_methods[num_descriptor](query_descriptor, db_descriptor)
        num_descriptor += 1
    return scores


def remove_frame(img):
    """ Remove frame from painting (arbitrarly ~5% of the image on each side) -> Helps in getting better MAP """
    m = 0.05
    p1, p2 = int(img.shape[0]*m), int(img.shape[1]*m)
    img = img[p1:img.shape[0]-p1, p2:img.shape[0]-p2]
    return img, p1, p2

######
# Image descriptors
######

def celled_1dhist(img, cells=[12, 12]):
    """ Divide image into cells and compute the 1d histogram. 
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    for i in range(cells[0]):
        for j in range(cells[1]):
            for ch in range(3):
                vals = np.histogram(img[w_ranges[i]:w_ranges[i+1], h_ranges[i]:h_ranges[i+1], ch], bins=np.arange(255))[0]
                normalized_hist = vals/vals.sum()
                descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def celled_2dhist(img, cells=[16, 16]):
    """ Divide image in cells and compute the 2d histogram in another color space.
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    img, p1, p2 = remove_frame(img)

    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    for i in range(cells[0]):
        for j in range(cells[1]):
            cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 1].reshape(-1)
            cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 10), np.arange(0, 255, 10)))[0]
            normalized_hist = vals/vals.sum()
            descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def celled_2dhist(img, cells=[16, 16]):
    """ Divide image in cells and compute the 2d histogram in another color space.
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    img, p1, p2 = remove_frame(img)

    descriptor = []
    w,h = img.shape[:2]
    w_ranges = [(i*w)//cells[0] for i in range(cells[0])]+[-1]
    h_ranges = [(i*h)//cells[1] for i in range(cells[1])]+[-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    for i in range(cells[0]):
        for j in range(cells[1]):
            cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 1].reshape(-1)
            cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 2].reshape(-1)
            vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 10), np.arange(0, 255, 10)))[0]
            normalized_hist = vals/vals.sum()
            descriptor.append(normalized_hist)
    
    # Score between 0 and max(measure(d1, d2))
    descriptor = np.array(descriptor).reshape(-1)
    descriptor = descriptor/(cells[0]*cells[1])
    return descriptor

def celled_2dhist_multiresolution(img, cells=[[6,6],[9,9]]):
    """ Divide image in cells and compute the 2d histogram in another color space.
            cells: Cell grid size (divides the image into [nxm] cells if cells=[n,m])
        returns: Image descriptor (np.array)
    """
    img, p1, p2 = remove_frame(img)

    descriptor = []
    w,h = img.shape[:2]
    for cell in cells:
        w_ranges = [(i*w)//cell[0] for i in range(cell[0])]+[-1]
        h_ranges = [(i*h)//cell[1] for i in range(cell[1])]+[-1]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
        for i in range(cell[0]):
            for j in range(cell[1]):
                cr = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 1].reshape(-1)
                cb = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1], 2].reshape(-1)
                vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 10), np.arange(0, 255, 10)))[0]
                normalized_hist = vals/vals.sum()
                descriptor.append(normalized_hist)
    
    return np.array(descriptor).reshape(-1)

def oned_hist(img):
    """ One dimensional histogram of images. 
        returns: Image descriptor (np.array)
    """
    descriptor = []

    if len(img.shape) == 3:
        img = img.reshape(img.shape[0]*img.shape[1], 3)
    img = img[np.newaxis, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    img_part = img
    cr = img_part[:, :, 1].reshape(-1)
    cb = img_part[:, :, 2].reshape(-1)
    vals = np.histogram(cr, bins=(np.arange(0, 255, 20), np.arange(0, 255, 20)))[0]
    normalized_hist = vals/vals.sum()
    vals2 = np.histogram(img_part[:, :, 1], bins=(np.arange(0, 255, 20)))[0]
    normalized_hist2 = vals2/vals2.sum()
    descriptor.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
    
    return np.array(descriptor).reshape(-1)

def twod_hist(img):
    """ Two dimensional histogram of images. 
        returns: Image descriptor (np.array)
    """
    descriptor = []
    if len(img.shape) == 3:
        img = img.reshape(img.shape[0]*img.shape[1], 3)
    img = img[np.newaxis, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb) 

    img_part = img
    cr = img_part[:, :, 1].reshape(-1)
    cb = img_part[:, :, 2].reshape(-1)
    vals = np.histogram2d(cr, cb, bins=(np.arange(0, 255, 5), np.arange(0, 255, 5)))[0]
    normalized_hist = vals/vals.sum()
    vals2 = np.histogram(img_part[:, :, 0], bins=(np.arange(0, 255, 5)))[0]
    normalized_hist2 = vals2/vals2.sum()
    descriptor.append(np.concatenate([normalized_hist.reshape(-1), normalized_hist2.reshape(-1)]))
    
    return np.array(descriptor).reshape(-1)

def mrhm(img, mask=None, num_blocks=16):
    """ Two dimensional histogram of images. 
        returns: Image descriptor (np.array)
    """
    
    if mask is not None:
        mask = (mask!=0).astype(np.uint8)*255
        x,y,w,h = 0,0,img.shape[1],img.shape[0]
        block_h = h // num_blocks #int(np.ceil(h / num_blocks))
        block_w = w // num_blocks #int(np.ceil(w / num_blocks))
    else:
        x,y = 0,0
        h,w = img.shape[:2]
        block_h = h // num_blocks #int(np.ceil(h / num_blocks))
        block_w = w // num_blocks #int(np.ceil(w / num_blocks))
        
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    w_ranges = [(i*w)//num_blocks for i in range(num_blocks)]+[-1]
    h_ranges = [(i*h)//num_blocks for i in range(num_blocks)]+[-1]

    features = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            image_block = img[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1]]
            if mask is not None:
                mask_block = mask[w_ranges[i]:w_ranges[i+1], h_ranges[j]:h_ranges[j+1]]
            else:
                mask_block = None

            block_feature = cv2.calcHist([image_block],[0,1,2], mask_block, [2,24,24], [0, 256, 0, 256, 0, 256])
            block_feature = cv2.normalize(block_feature, block_feature)
            features.extend(block_feature.flatten())

    return np.stack(features).flatten()
    

OPTIONS = ["onedcelled", "CBHC", "1dhist", "2dhist", "CBHCM"]

METHOD_MAPPING = {
    OPTIONS[0]: celled_1dhist,
    OPTIONS[1]: celled_2dhist,
    OPTIONS[2]: oned_hist,
    OPTIONS[3]: twod_hist,
    OPTIONS[4]: celled_2dhist_multiresolution
}

def get_method(method):
    return METHOD_MAPPING[method]


#######
TC = "text"
HC = "HC"

OPTIONS = [TC, HC]

METHOD_MAPPING_EXTR = {
    OPTIONS[0]: TC,
    OPTIONS[1]: celled_2dhist # mrhm
}

def get_descriptor_extractor(method):
    return METHOD_MAPPING_EXTR[method]

