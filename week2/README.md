# Week 1

## Pre-requisites
- OpenCV
- Numpy
- Scipy
- tqdm

You can install these packages with pip using: 
`pip install requirements.txt`

IMPORTANT: It seems that the text_boxes.pkl from the first query dataset is in the wrong format. To fix it run the script in db_utils/change_format_text_boxes.py:
`python db_utils/change_format_text_boxes.py <path_to_pkl_file>`. You should also rotate the 15th image of the query set 2.

## Instructions
`python match\_paintings.py <path_to_db> (e.g., /home/sergio/MCV/M1/DB) <db_folder_name> (e.g., BBDD) <query_set_folder_name> (e.g., qsd1_w1) --masking <0 apply mask / 1 don't> -rm <name of the retrieval method> -mm <name of the masking method> -tm <name of the text detection method> --output_pkl <path to save pkl file with query assignments> --output_mask <folder to save mask images> --opkl_text <path to save pkl file with text bounding boxes>`

## List of methods
<ul>
    <li> rm: [CBHC, 2dhist, CBHCM]</li>
    <li> mm: [CBHS, PBM]</li>
    <li> tm: [MM, MM2]</li>
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