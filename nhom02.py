import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import histogram, vector, downsample
import statistics as st
import random
from tqdm import tqdm

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28).astype(np.float64)
    
    return images, labels

X_train, y_train = load_mnist('Data/', kind='train')
print('Train images shape:', X_train.shape)
print('Train labels shape:', y_train.shape)

X_test, y_test = load_mnist('Data/', kind='t10k')
print('Test images shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print()

print('Extracting vector...')
X_train_vector = vector.get_vector(X_train)
X_test_vector = vector.get_vector(X_test)
print('Train vector shape:', X_train_vector.shape)
print('Test vector shape:', X_test_vector.shape)
print()

print('Extracting downsample...')
X_train_downsample = downsample.get_downsample(X_train)
X_test_downsample = downsample.get_downsample(X_test)
print('Train downsample shape:', X_train_downsample.shape)
print('Test downsample shape:', X_test_downsample.shape)
print()

print('Extracting histogram...')
X_train_histogram = histogram.get_histogram(X_train)
X_test_histogram = histogram.get_histogram(X_test)
print('Train histogram shape:', X_train_histogram.shape)
print('Test histogram shape:', X_test_histogram.shape)
print()


def kNN_predict(X_train, y_train, predict_sample, k):
    squared_diffs = np.sum((X_train - predict_sample)**2, axis=1)
    distances = np.sqrt(squared_diffs)
    idx = np.argpartition(distances, k)[:k]
    labels = y_train[idx]
    answer = st.mode(labels)
    return answer

def predict_downsample(predict_sample, k):
    return kNN_predict(X_train_downsample, y_train, predict_sample, k)

def predict_vector(predict_sample, k):
    return kNN_predict(X_train_vector, y_train, predict_sample, k)

def predict_histogram(predict_sample, k):
    return kNN_predict(X_train_histogram, y_train, predict_sample, k)

num_test = 10000
k_lock = 7

predictions = np.empty(num_test)
print('Measuring accuracy of vectorized...')
for i in tqdm(range(num_test)):
    predictions[i] = predict_vector(X_test_vector[i], k_lock)
accuracy = np.mean(predictions == y_test[:num_test])
print(f"Accuracy of vectorized (k = {k_lock}):", accuracy, end='\n\n')

predictions = np.empty(num_test)
print('Measuring accuracy of downsample...')
for i in tqdm(range(num_test)):
    predictions[i] = predict_downsample(X_test_downsample[i], k_lock)
accuracy = np.mean(predictions == y_test[:num_test])
print(f"Accuracy of downsample (k = {k_lock}):", accuracy, end='\n\n')

predictions = np.empty(num_test)
print('Measuring accuracy of histogram...')
for i in tqdm(range(num_test)):
    predictions[i] = predict_histogram(X_test_histogram[i], k_lock)
accuracy = np.mean(predictions == y_test[:num_test])
print(f"Accuracy of histogram (k = {k_lock}):", accuracy, end='\n\n')

# sample_count = 10
# nrows = 2
# ncols = sample_count // nrows
# random_indexes = random.sample(range(X_test.shape[0]), sample_count)
# predictions = np.empty(sample_count)

# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i, test_index in enumerate(random_indexes):
#     img = X_test[test_index]
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#     predictions[i] = predict(X_test_downsample[test_index])

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# print("Predictions: ", predictions)
# plt.show()
