# Week 1

## Pre-requisites
- OpenCV
- Numpy
- Scipy
- tqdm
- scikit-image
- textdistance
- pytesseract

You can install these packages with pip using: 
`pip install requirements.txt`

IMPORTANT: It seems that the text_boxes.pkl from the first query dataset is in the wrong format. To fix it run the script in db_utils/change_format_text_boxes.py:
`python3 db_utils/change_format_text_boxes.py /{partial_path}/qsd1_w2/text_boxes.pkl`

You should also rotate the 15th image of the query set 2.

#### Installing Tesseract
In order to run the code, you should have Tesseract installed in your OS.
You can visit [this link](https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i)
which includes instructions to install it in most common used systems.

## Instructions
`python match\_paintings.py <path_to_db> (e.g., /home/sergio/MCV/M1/DB) <db_folder_name> (e.g., BBDD) <query_set_folder_name> (e.g., qsd1_w1) --masking <0 don't / 1 apply mask> -rmm <list of retrieval methods (1 or more)> -mm <name of the masking method> -tm <name of the text detection method> --matching_measures <measures to use to compare the descriptor of each method (1 or more)> --output_pkl <path to save pkl file with query assignments> --output_mask <folder to save mask images> --opkl_text <path to save pkl file with text bounding boxes>`

## List of methods
<ul>
    <li> matching_measures: [l1_dist, l2_dist, hellinger, js_div, levenshtein, jaro_winkler, ratcliff_obershelp]</li>
    <li> rmm: [text, HC, HOG, DCT, LBP]</li>
    <li> mm: [ES, PBM]</li>
    <li> tm: [MM, MM2, MM3, MM4]</li>
    <li> filter_type: [median]</li>
    
</ul>

## List of measures
<ul>
    <li> Text comparison measures: [levenshtein, jaro_winkler, ratcliff_obershelp] </li>
    <li> Histogram comparison measures: [l1_dist, l2_dist, hellinger, js_div] </li>
</ul>

## Notes
 - The folder should contain the database folder aswell as the query set. \
    e.g., \
    -db_folder \ (<- <path_to_db> parameter (absolute path))
    -- BBDD \ (<- <db_folder_name> parameter (relative path with respect to db_folder -> it is only necessary to indicate the folder name))
    -- qsd2_w1 (<- <query_set_folder_name> parameter (relative path with respect to db_folder -> it is only necessary to indicate the folder name))

Method reference list:
- CBHC: Cell-based histogram comparison
- CBHS: Cell-based histogram segmentation
- CBHCM: Multiresolution cell-based histogram comparison
- PBM: Probability-based masking
- 2dhist: Color histogram (2D luminance + 1D luminance histograms -> Descriptor)
- MM: Morphology-based text detector 1
- MM2: Morphology-based text detector 2
