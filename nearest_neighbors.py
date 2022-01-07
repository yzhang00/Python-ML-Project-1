# PROBLEM 2: N NEAREST NEIGHBORS CLASSIFICATION

from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import numpy

# loading training and test data
(X_train, train_y), (X_test, test_y) = mnist.load_data()

# flattening 28 x 28 element 2d array into 1d array to match knn param requirements
train_X = []
test_X = []

for i in range(len(X_train)):
    train_X.append(X_train[i].flatten())

for i in range(len(X_test)):
    test_X.append(X_test[i].flatten())

# beginning n nearest neighbor classification
print('N NEAREST NEIGHBORS CLASSIFIER')
n_neighbors = [1, 5, 10, 20, 50, 100]
train_error = []
test_error = []

# calculating training and test error rate for each value of n
for i in n_neighbors:
    # initializing knn classifier with n = i neighbors
    neigh = KNeighborsClassifier(n_neighbors=i)
    print("NUM NEIGHBORS: %d" % i)
    # calculating and storing training error rate for n = i neighbors
    y_train_pred = neigh.fit(train_X, train_y).predict(train_X)
    print("TRAINING DATA - Num. mislabeled points: %d / %d" %
          ((train_y != y_train_pred).sum(), len(train_X)))
    train_error.append((train_y != y_train_pred).sum() / len(train_X))
    # calculating and storing test error rate for n = i neighbors
    y_test_pred = neigh.fit(train_X, train_y).predict(test_X)
    print("TEST DATA - Num. mislabeled points: %d / %d" %
          ((test_y != y_test_pred).sum(), len(test_X)))
    test_error.append((test_y != y_test_pred).sum() / len(test_X))

# plotting training and test error rates for each value of n
pyplot.plot(n_neighbors, train_error, label="training data error rate")
pyplot.plot(n_neighbors, test_error, label="test data error rate")
pyplot.xlabel("N Neighbor Values")
pyplot.ylabel("Error Rate")
pyplot.title("N Neighbor Values vs. Error Rate for Training and Test Data")
pyplot.legend()
pyplot.show()
