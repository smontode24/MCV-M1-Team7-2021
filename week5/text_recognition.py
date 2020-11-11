import pytesseract
import debug_utils
import cv2
from debug_utils import *
import os

def extract_text_from_imgs(imgs_list, text_bboxes):
    """ Recognize text in images. 
        params:
            imgs_list = [[img1_1, img1_2], [img2], [img3, img3_2], [img4]]
            text_bboxes: [[[2,3,5,6],[6,6,83,23]], [[3,4,5,6]], ...]
        returns:
            text_results = [["Isaac Pérez", "Lena"], ["Sergio"], ...]
    """ 
    text_results = []
    img_n = 0
    for paintings, painting_bboxes in zip(imgs_list, text_bboxes):
        text_paintings = []
        for painting, bbox in zip(paintings, painting_bboxes):
            res_string = img_w_mask_to_string(painting, bbox)
            text_paintings.append(res_string)
        text_results.append(text_paintings)
        img_n += 1

    if isDebug():
        for paintings, painting_bboxes, texts_paintings in zip(imgs_list, text_bboxes, text_results):
            text_paintings = []
            for painting, bbox, text in zip(paintings, painting_bboxes, texts_paintings):
                painting_copy = painting.copy()
                painting_copy = cv2.rectangle(painting_copy, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0 ,0), 10)
                cv2.imshow("result text segm", cv2.resize(painting_copy,(512,512)))
                print("Obtained text:", text)
                cv2.waitKey(0)
    
    return text_results

def img_w_mask_to_string(img, bbox):
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #img = cv2.medianBlur(img, 3)
    if img.shape[0] != 0:
        result = pytesseract.image_to_string(img)
    else:
        result = ""
    return take_largest_name(result)
    
def take_largest_name(result):
    """ Refine text detection """
    splits = result.split("\n")
    largest_name = ""
    for split in splits:
        if len(split) > len(largest_name):
            largest_name = split
    return largest_name