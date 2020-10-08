import numpy as np
from scipy.spatial.distance import cdist
""" Here create metrics: l2, l1, similarity metrics,... """

def l2_dist(m1, m2):
    """ m1 (mxd), m2 (nxd) => similarities (mxn) (-distance) """
    x2 = np.sum(m1**2, axis=1, keepdims=True)
    y2 = np.sum(m2**2, axis=1)
    xy = np.dot(m1, m2.T)
    dist = np.sqrt(x2 - 2*xy + y2)
    return dist # Most similar are bigger

def l1_dist(m1, m2):
    """ m1 (mxd), m2 (nxd) => similarities (mxn) (-distance) """
    return cdist(m1, m2, 'minkowski', p=1) # Most similar are bigger