import numpy as np
from evaluation.mask_evaluation import performance_evaluation_window
import matplotlib.pyplot as plt

"""
MAP metric from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
"""

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def show_f1_scores(top_k_matches, num_matches, gts, max_matches = 100):
    nms = []
    f1s = []

    for nm in np.arange(max_matches):
        top_k_matches = np.argpartition(num_matches, [-1])[:,-1:][:,::-1]
        for i in range(len(top_k_matches)):
            if num_matches[i, top_k_matches[i][0]] < nm:
                top_k_matches[i][0] = -1
        f1s.append(f1_id_in_db(top_k_matches, gts))
        nms.append(nm)

    plt.plot(nms, f1s)
    print("best threshold:", np.argmax(f1s))
    plt.ylim((0,1))
    plt.show()

def f1_id_in_db(predicted, gts):
    assignment_in_db = np.array([bool(gt[0] != -1) for gt in gts])
    predicted_in_db = np.array([bool(pred[0] != -1) for pred in predicted])
    TP = (assignment_in_db == predicted_in_db).astype(np.uint8).sum() 
    FP = (np.logical_not(assignment_in_db) == predicted_in_db).astype(np.uint8).sum() 
    FN = (assignment_in_db == np.logical_not(predicted_in_db)).astype(np.uint8).sum()
    f1 = performance_evaluation_window(TP, FN, FP)[3]
    return f1