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
    <li> rm: [onedcelled, twodcelled, twodcelled2, 1dhist, 2dhist]</li>
    <li> mm: [cbhs]</li>
</ul>

## Notes
 - The folder should contain the database folder aswell as the query set with the original names. \
    e.g., \
    -db_folder \
    -- BBDD \
    -- qsd1_w1
