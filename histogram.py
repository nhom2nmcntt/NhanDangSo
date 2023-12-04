import numpy as np;

def get_histogram(data):
    Shape = data.shape;
    ret = np.zeros((256), dtype = int);
    for i in range(0, Shape[0]):
        for j in range(0, 28):
            for k in range(0, 28):
                ret[int(data[i][j][k])] += 1;
    
    return ret;