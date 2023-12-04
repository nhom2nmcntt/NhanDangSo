import numpy as np

def get_downsample(X_train):

    a, b = X_train.shape

    b_downsampled = b // 2

    reshaped_array = X_train[:, :b_downsampled * 2].reshape(a, b_downsampled, 2)

    downsampled_array = reshaped_array.mean(axis=2)
    output_array = np.reshape(downsampled_array, downsampled_array(.shape[0], -1))

    return output_array
