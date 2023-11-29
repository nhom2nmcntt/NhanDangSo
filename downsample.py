import numpy as np

def get_downsample(data):
    data_shape = data.shape
    sampleCount = data_shape[0]
    res = np.empty((0, 0))
    return res