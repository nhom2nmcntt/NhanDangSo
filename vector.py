import numpy as np

def get_vector(X_train):
    output_array = np.reshape(X_train, (X_train.shape[0], -1))
    return output_array