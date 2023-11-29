import numpy as np
from collections import Counter

def get_histogram(data):
    data_shape = data.shape
    sampleCount = data_shape[0]
    res = np.empty((0, 0))
    return res