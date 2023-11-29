import numpy as np
from collections import Counter

def get_histogram(data):
    data_shape = data.shape
    sampleCount = data_shape[0]
    res = np.empty((sampleCount, 256))
    image_size = data_shape[1] * data_shape[2]
    for sampleIndex in range(sampleCount):
        counter = Counter(data[sampleIndex].reshape(-1))
        for j in range(256):
          res[sampleIndex, j] = counter[j] / (image_size)

    return res