import numpy as np
from scipy.spatial.distance import cdist
import cv2
from debug_utils import *

""" 
All these metrics receive two matrices: m1 (mxd), m2 (nxd)
And returns => distance (mxn) [lower -> more similar]  
"""

def l2_dist(m1, m2):
    """ L2 distance """
    x2 = np.sum(m1**2, axis=1, keepdims=True)
    y2 = np.sum(m2**2, axis=1)
    xy = np.dot(m1, m2.T)
    dist = np.sqrt(x2 - 2*xy + y2)
    return dist

def l1_dist(m1, m2):
    """ L1 distance """
    return cdist(m1, m2, 'minkowski', p=1) 

def hellinger_kernel(m1, m2):
    """ Hellinger kernel """
    return -np.sqrt(np.dot(m1, m2.T)) # Negative value because we want that lower is better

def js_div(m1, m2):
    """ Jensen-Shannon divergence """
    result = []

    m1 += 1e-9
    m2 += 1e-9
    
    m1 = m1/m1.sum(axis=1, keepdims=True)
    m2 = m2/m2.sum(axis=1, keepdims=True)
    for i in range(m1.shape[0]):
        kl_m1_m2 = (m1[i,:]*np.log(m1[i,:]/m2)).sum(axis=1)
        kl_m2_m1 = (m2*np.log(m2/m1[i,:])).sum(axis=1)
        result.append((kl_m1_m2+kl_m2_m1)/2)
    
    return np.array(result)

# Comparing two strings by counting minimum number of operations
# required to transform one string to another
# seq1 & seq2 are strings to be compared 
def levensthein(seq1, seq2): # TODO: Substituting characters should not be as expensive as adding/removing
    #SOURCE: https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )

    #print (matrix)
    # RETURNS Nº of changes required.
    # Between "mama" and "papa" == 2
    return (matrix[size_x - 1, size_y - 1])

MEASURES = {
    "l2_dist": l2_dist,
    "l1_dist": l1_dist,
    "hellinger": hellinger_kernel,
    "js_div": js_div,
    "levensthein": levensthein
}

def get_measure(name):
    return MEASURES[name]