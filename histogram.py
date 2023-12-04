import numpy as np
import matplotlib.pyplot as plt
def get_histogram(arr_3d):
    a, b, c = arr_3d.shape
    flattened_arr = arr_3d.reshape((a, -1))
    hist, edges = np.histogram(flattened_arr, bins=b*c, density=True)
    arr_2d = np.column_stack((edges[:-1], hist))

    return arr_2d
