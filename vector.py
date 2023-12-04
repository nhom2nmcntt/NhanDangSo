import numpy as np


def get_vector(arr_3d):
    a, b, c = arr_3d.shape
    arr_2d = arr_3d.reshape(a, -1)
    return arr_2d
