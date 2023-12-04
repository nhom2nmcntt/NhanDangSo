import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import vectorize as vtr
import downsample as ds 
import histogram as htg
import standardize

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

X_t10k, y_t10k = load_mnist('Data/', kind='t10k')
print('Test images shape: ', X_t10k.shape)
print('Test labels shape: ', y_t10k.shape)
print()

std_Xtrain = standardize.stdize(X_train);
std_Xtest = standardize.stdize(X_t10k);

VectorTrainImages = vtr.get_vector(std_Xtrain);
VectorTestImages = vtr.get_vector(std_Xtest);
print(VectorTrainImages.shape);
print(VectorTestImages.shape);
print();


DownSampleTrainImages = ds.get_downsample(standardize.stdize(std_Xtrain));
DownSampleTestImages = ds.get_downsample(standardize.stdize(std_Xtest));
print(DownSampleTrainImages.shape);
print(DownSampleTestImages.shape);
print();


TrainImagesHistogram = htg.get_histogram(X_train);
TestImagesHistogram = htg.get_histogram(X_t10k);
print(TrainImagesHistogram.shape);
print(TestImagesHistogram.shape);


#fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
#ax = ax.flatten()
#for i in range(10):
#    img = X_train[y_train == i][0]
#    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
#ax[0].set_xticks([])
#ax[0].set_yticks([])
#plt.tight_layout()
#plt.show()