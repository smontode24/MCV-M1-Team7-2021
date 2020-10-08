import numpy as np

""" TODO: Painting region masking:
- Color histograms
- Locate countours (high x and y gradients)
- Morphological operations to fill holes / remove unnecesary isolated parts
- ...
"""

def bg_mask(query_imgs): # TODO
    """ Obtain mask from image """
    # Temporary mask (grab all the painting part)
    return [np.ones((img.shape[0], img.shape[1])) for img in query_imgs]