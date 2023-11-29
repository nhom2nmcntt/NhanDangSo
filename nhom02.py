import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
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

for i in range(28):
    for j in range(28):
        print('{0: <5}'.format(X_train[0][i][j]), end=' ')
    print()
    

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()