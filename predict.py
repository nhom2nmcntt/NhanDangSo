import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import histogram, vector, downsample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import tqdm
import statistics as st
import random

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

def load_and_display():
    X_train, y_train = load_mnist('Data/', kind='train')
    print('Train images shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)

    X_test, y_test = load_mnist('Data/', kind='t10k')
    print('Test images shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print()

    return X_train, y_train, X_test, y_test

def extract_vector(X_train, X_test):
    print('Extracting vector...')
    X_train_vector = vector.get_vector(X_train)
    X_test_vector = vector.get_vector(X_test)
    print('Train vector shape:', X_train_vector.shape)
    print('Test vector shape:', X_test_vector.shape)
    print()
    return X_train_vector, X_test_vector

def extract_downsample(X_train, X_test):
    print('Extracting downsample...')
    X_train_downsample = downsample.get_downsample(X_train)
    X_test_downsample = downsample.get_downsample(X_test)
    print('Train downsample shape:', X_train_downsample.shape)
    print('Test downsample shape:', X_test_downsample.shape)
    print()
    return X_train_downsample, X_test_downsample

def extract_histogram(X_train, X_test):
    print('Extracting histogram...')
    X_train_histogram = histogram.get_histogram(X_train)
    X_test_histogram = histogram.get_histogram(X_test)
    print('Train histogram shape:', X_train_histogram.shape)
    print('Test histogram shape:', X_test_histogram.shape)
    print()
    return X_train_histogram, X_test_histogram

def show_a_sample(sample):
    fig, ax = plt.subplots()
    ax.imshow(sample, cmap='Greys', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

'''
    https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
    https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    https://www.geeksforgeeks.org/how-to-calculate-the-mode-of-numpy-array/
'''
def kNN_predict(X_train, y_train, predict_sample_downsampled):
    def distance(sample1):
        sum_sq = np.sum(np.square(sample1 - predict_sample_downsampled))
        return np.sqrt(sum_sq)

    distances = np.array(list(map(distance, X_train)))
    k = 30
    idx = np.argpartition(distances, k)[:k]
    labels = y_train[idx]
    answer = st.mode(labels)
    return answer

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_display()
    X_train_downsample, X_test_downsample = extract_downsample(X_train, X_test)
    def predict(predict_sample):
        return kNN_predict(X_train_downsample, y_train, predict_sample)

    # predictions = np.empty(400)
    # for i in tqdm(range(400)):
    #     predictions[i] = predict(X_test_downsample[i])
    # accuracy = np.mean(predictions == y_test[:400])
    # print('Accuracy:', accuracy)
    sample_count = 10
    nrows = 2
    ncols = sample_count // nrows
    random_indexes = random.sample(range(X_test.shape[0]), sample_count)
    predictions = np.empty(sample_count)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i, test_index in enumerate(random_indexes):
        img = X_test[test_index]
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        predictions[i] = predict(X_test_downsample[test_index])

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    print("Predictions: ", predictions)
    plt.show()

# def train_kNN_histogram(X_train_histogram, y_train, X_test_histogram, y_test):
#     k = 9
#     # Accuracy: {1: 0.3028, 
#     #            2: 0.2979, 
#     #            3: 0.309, 
#     #            4: 0.318, 
#     #            5: 0.323, 
#     #            6: 0.3212, 
#     #            7: 0.3249, 
#     #            8: 0.324, 
#     #            9: 0.3265}
#     kNN = KNeighborsClassifier(n_neighbors=k)
#     kNN.fit(X_train_histogram, y_train)
#     predictions = kNN.predict(X_test_histogram)
#     accuracy = np.mean(predictions == y_test)
#     print(f"Accuracy: {accuracy}")
#     return kNN

# def train_kNN_downsample(X_train_downsample, y_train, X_test_downsample, y_test):
#     k = 7
#     # 1: 0.9717
#     # 2: 0.968
#     # 3: 0.9731
#     # 4: 0.9739
#     # 5: 0.9735
#     # 6: 0.9737
#     # 7**: 0.974
#     # 8: 0.973
#     # 9: 0.9727
#     # 10: 0.972
#     kNN = KNeighborsClassifier(n_neighbors=k)
#     kNN.fit(X_train_downsample, y_train)
#     predictions = kNN.predict(X_test_downsample)
#     accuracy = np.mean(predictions == y_test)
#     print(f"Accuracy: {accuracy}")
#     return kNN

# def train_logistic_reg_downsample(X_train_downsample, y_train, X_test_downsample, y_test):
#     logistic_reg = LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='ovr')
#     logistic_reg.fit(X_train_downsample, y_train)
#     y_pred = logistic_reg.predict(X_test_downsample)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     return logistic_reg


# def get_kNN_histogram_model():
#     X_train, y_train, X_test, y_test = load_and_display()
#     X_train_histogram, X_test_histogram = extract_histogram(X_train, X_test)
#     kNN = train_kNN_histogram(X_train_histogram, y_train, X_test_histogram, y_test)
#     return kNN

# def get_kNN_downsample_model():
#     X_train, y_train, X_test, y_test = load_and_display()
#     show_train_samples(X_train, y_train)
#     X_train_downsample, X_test_downsample = extract_downsample(X_train, X_test)
#     kNN = train_kNN_downsample(X_train_downsample, y_train, X_test_downsample, y_test)
#     return kNN

# def get_logistic_reg_downsample_model():
#     X_train, y_train, X_test, y_test = load_and_display()
#     show_train_samples(X_train, y_train)
#     X_train_downsample, X_test_downsample = extract_downsample(X_train, X_test)
#     logistic_reg = train_logistic_reg_downsample(X_train_downsample, y_train, X_test_downsample, y_test)
#     return logistic_reg

def show_a_train_sample(sample):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,)
    ax[0].imshow(sample, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def show_train_samples(X_train, y_train):
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0]
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


# def predict(model, sample_image_array):
#     ans = model.predict(sample_image_array)
#     return ans

# if __name__ == "__main__":
#     logistic_reg = get_logistic_reg_downsample_model()
#     print("Hello world!")




