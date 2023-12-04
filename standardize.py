import numpy as np

def stdize(data):
    Shape = data.shape;
    ret = data;
    for i in range(0, Shape[0]):
        for j in range(0, 28):
            for k in range(0, 28):
                ret[i][j][k] = ret[i][j][k] / 255.0;

    return ret;
