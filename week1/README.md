# Week 1

## Pre-requisites
- OpenCV
- Numpy
- Scipy

You can install these packages with pip using: 
`pip install requirements.txt`

## Instructions
`python match\_paintings.py <path_to_db> (e.g., /home/sergio/MCV/M1/DB) <db_folder_name> (e.g., BBDD) <query_set_folder_name> (e.g., qsd1_w1) --masking <0 apply mask / 1 don't> -rm <name of the retrieval method> -mm <name of the masking method> --output_pkl <path to save pkl file with query assignments>        --output_mask <folder to save mask images>`

## List of methods
<ul>
    <li> rm: [onedcelled, CBHC, twodcelled2, 1dhist, 2dhist]</li>
    <li> mm: [CBHS, PBM]</li>
</ul>

## Notes
 - The folder should contain the database folder aswell as the query set with the original names. \
    e.g., \
    -db_folder \
    -- BBDD \
    -- qsd1_w1

Method reference list:
- CBHC: Cell-based histogram comparison
- CBHS: Cell-based histogram segmentation
- PBM: Probability-based masking
- 2dhist: Color histogram (2D luminance + 1D luminance histograms -> Descriptor)