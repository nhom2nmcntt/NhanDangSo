import numpy as np

def get_vector(X_train):
    output_array = np.reshape(X_train, (X_train.shape[0], -1))
    
    return output_array

try:
    with open("X_train.txt", "r") as file:

        X_train = np.array([list(map(int, line.split())) for line in file.readlines()])
except FileNotFoundError:
    print("File X_train.txt not found.")
