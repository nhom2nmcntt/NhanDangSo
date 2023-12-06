import numpy as np
from collections import Counter

def get_histogram(data):
    data_shape = data.shape
    sampleCount = data_shape[0]
    # tao mang rong
    res = np.empty((sampleCount, 256)) 
    # kich thuoc anh
    image_size = data_shape[1] * data_shape[2]
    for sampleIndex in range(sampleCount):
        # dung counter de dem
        counter = Counter(data[sampleIndex].reshape(-1))
        for j in range(256):
          res[sampleIndex, j] = counter[j] / (image_size)

    return res
