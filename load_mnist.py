# TEST FILE: this file tests loading/formatting mnist data

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot
import numpy

# loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# plotting
# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()

print('len X_train: ' + str(len(train_X)))
print('len X_train[0]: ' + str(len(train_X[0])))
print('len X_train[0][0]: ' + str(len(train_X[0][0])))

x2 = []
for i in range(600):
    x2.append(train_X[i].flatten())

print('len x2: ' + str(len(x2)))
print('len x2[0]: ' + str(len(x2[0])))
print('len x2[0][0]: ' + str(len(x2[0][0])))
