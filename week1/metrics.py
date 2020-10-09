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

def l1_dist_median(m1, m2):
    total = []
    
    step = m1.shape[1]//100
    for i in range(99):
        total.append(cdist(m1[:,step*i:step*(i+1)], m2[:,step*i:step*(i+1)], 'minkowski', p=1))
    total.append(cdist(m1[:,step*(i+1):], m2[:,step*(i+1):], 'minkowski', p=1))
    total = np.array(total)
    res = np.mean(total, 0)
    return res

def hellinger_kernel(m1, m2):
    """ m1 (mxd), m2 (nxd) => similarities (mxn) (-distance) """
    return -np.sqrt(np.dot(m1, m2.T))

def js_div(m1, m2):
    result = []

    m1 += 1e-9
    m2 += 1e-9
    
    m1 = m1/m1.sum(axis=1, keepdims=True)
    m2 = m2/m2.sum(axis=1, keepdims=True)
    for i in range(m1.shape[0]):
        for j in range(3): #TODO
            kl_m1_m2 = (m1[i,:]*np.log(m1[i,:]/m2)).sum(axis=1)
            kl_m2_m1 = (m2*np.log(m2/m1[i,:])).sum(axis=1)
            result.append((kl_m1_m2+kl_m2_m1)/2)
    
    return np.array(result)