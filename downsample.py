import numpy as np;

def get_downsample(data):
    Shape = data.shape;
    ret = np.zeros((Shape[0], 196));
    idx = -1;
    for i in range(0, Shape[0]):
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                if(idx + 1 < 196): idx = idx + 1;
                ret[i][idx] = (data[i][j][k] + data[i][j][k + 1] + data[i][j + 1][k] + data[i][j + 1][k + 1]) / 4.0;

    return ret;
