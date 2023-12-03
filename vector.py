import numpy as np

def get_vector(data):
    data_shape = data.shape
    sampleCount = data_shape[0]
    res = np.empty((sampleCount, 784))
    for sampleIndex in range(sampleCount):
        for i in range(data_shape[1]):
            for j in range(data_shape[2]):
                #Chuyen mang 2 chieu thanh mang 1 chieu
                res[sampleIndex, 28*i + j] = data[sampleIndex, i, j]
    return res