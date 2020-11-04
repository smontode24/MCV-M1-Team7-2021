import cv2
import numpy as np
from scipy.ndimage import filters

def harris_respone (im, sigma=3):
    """Harris corner detector response function for each pixel in greylevel images"""

    #Derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imy)

    #Components of the Harris Matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    #Determinant and Trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    #This gives an image with each pixel containing the value of the Harris response function.
    return Wdet/Wtr

def harris_points(harrisim, min_dist =10, threshold = 0.1):
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """

    # Find top corner candidates above a threshold

    corner_threshold = harrisim.max() * threshold
    harrisim_t =(harrisim > corner_threshold) * 1

    #Get coordinates of candidates and values

    coordinates = np.array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0], c[1]] for c in coordinates]
    index = np.argsort(candidate_values)

    #Store located points
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    #Select best points takind min distance into account
    filtered_coordinates = []
    for i in index:

        if allowed_locations[coordinates[i, 0], coordinates[i, 1]] ==1:
            filtered_coordinates.append(coordinates[i])
            allowed_locations[(coordinates[i, 0]-min_dist):(coordinates[i, 0]+min_dist), (coordinates[i, 1]-min_dist):(coordinates[i, 1]+min_dist)] = 0

    return filtered_coordinates

def descriptor(image, filtered_coordinates, wid = 5):
    """ For each point return, pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """

    desc = []

    for coods in filtered_coordinates:
        patch = image[coods[0]-wid:coods[0]+wid+1, coods[1]-wid:coods[1]+wid+1].flatten()
        desc.append(patch)

    return desc

def match(desc1, desc2, threshold = 0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross-correlation. """

    n = len(desc1[0])

    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range (len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores

def match2side (desc1, desc2, threshold =0.5):
    """This function matches each descriptor to its best candidate in the other
        image using normalized cross-correlation."""
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12= np.where(matches_12 >= 0)[0]

    #Remove matches non-symetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12