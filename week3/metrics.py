import numpy as np
from scipy.spatial.distance import cdist
import cv2
from debug_utils import *
import textdistance         # src -> https://github.com/life4/textdistance

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

######
# Text distance metrics
######
def levensthein(query, db): # TODO: Substituting characters should not be as expensive as adding/removing -> why?
    # Normalized version of levensthein
    # query -> list of texts detected on the query images
    # db -> list of texts of the db 
    scores = np.zeros((len(query), len(db)))
    for i, q_text in enumerate(query):
        for j, db_text in enumerate(db):
            scores[i,j] = textdistance.levensthein.distance(q_text, db_text)
            scores[i,j] = scores[i,j] / max(len(db_text), len(db_text))
    return scores

def normalized_levensthein(seq1, seq2):
    """ List of text authors in seq1 and seq2 """
    scores = np.zeros((len(seq1), len(seq2)))
    for i, s1 in enumerate(seq1):
        for j, s2 in enumerate(seq2):
            if len(s1) == 0 and len(s2) == 0:
                scores[i, j] = 1
            else:
                scores[i, j] = levensthein(s1, s2)
                scores[i, j] = scores[i, j]/max(len(s1), len(s2))
    
def jaro_winkler(query, db): # also normalized
    scores = np.zeros((len(query), len(db)))
    for i, q_text in enumerate(query):
        for j, db_text in enumerate(db):
            scores[i,j] = textdistance.jaro_winkler.distance(q_text, db_text)
            scores[i,j] = scores[i,j] / max(len(db_text), len(db_text))
    return scores

def ratcliff_obershelp(query, db):
    # Returns a value between 0 and 1 -> usually 1 means the strings match but in this implementation it means the strings differ
    scores = np.zeros((len(query), len(db)))
    for i, q_text in enumerate(query):
        for j, db_text in enumerate(db):
            scores[i,j] = textdistance.ratcliff_obershelp.distance(q_text, db_text)
    return scores


MEASURES = {
    "l2_dist": l2_dist,
    "l1_dist": l1_dist,
    "hellinger": hellinger_kernel,
    "js_div": js_div,
    "levenshtein": normalized_levensthein,
    "jaro_winkler": jaro_winkler,
    #"smith_waterman": smith_waterman,
    "ratcliff_obershelp": ratcliff_obershelp,


}

def get_measure(name):
    return MEASURES[name]