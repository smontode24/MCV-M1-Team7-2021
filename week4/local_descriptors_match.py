import skimage
from skimage.feature import match_descriptors

def automatic_brute_force_match(descriptors1, descriptors2, metric="euclidean", max_ratio=0.8):
    matches = match_descriptors(descriptors1, descriptors2, metric=metric, max_ratio=max_ratio)
    return len(matches)