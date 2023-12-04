import numpy as np;

def get_vector(data):
    data_shape = data.shape;
    ret = np.zeros((data_shape[0], 784));
    for i in range(0, data_shape[0]):
        for j in range(0, 28):
            for k in range(0, 28):
                ret[i][j * 28 + k] = data[i][j][k];
    
    
    return ret;