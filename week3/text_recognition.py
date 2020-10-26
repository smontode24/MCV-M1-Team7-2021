import pytesseract
import debug_utils

def extract_text_from_imgs(imgs_list, text_bboxes):
    """ Recognize text in images. 
        params:
            imgs_list = [[img1_1, img1_2], [img2], [img3, img3_2], [img4]]
            text_bboxes: [[[2,3,5,6],[6,6,83,23]], [[3,4,5,6]], ...]
        returns:
            text_results = [["Isaac Pérez", "Lena"], ["Sergio"], ...]
    """ 
    text_results = []
    for paintings, painting_bboxes in zip(imgs_list, text_bboxes):
        text_paintings = []
        for painting, bbox in zip(paintings, painting_bboxes):
            text_paintings.append(img_w_mask_to_string(painting, bbox))
        text_results.append(text_paintings)
    return text_results

def img_w_mask_to_string(img, bbox):
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    result = pytesseract.image_to_string(img)
    return result
    