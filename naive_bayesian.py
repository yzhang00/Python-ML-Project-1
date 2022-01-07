# PROBLEM 1: NAIVE BAYESIAN CLASSIFICATION USING GAUSSIAN ASSUMPTION

from tensorflow.keras.datasets import mnist
from sklearn.naive_bayes import GaussianNB
import numpy

# loading training and test data
(X_train, train_y), (X_test, test_y) = mnist.load_data()

# initializing gaussian naive bayesian classifier
gnb = GaussianNB()
print('NAIVE BAYESIAN CLASSIFIER')

# flattening 28 x 28 element 2d array into 1d array to match gnb param requirements
train_X = []
test_X = []

for i in range(len(X_train)):
    train_X.append(X_train[i].flatten())

for i in range(len(X_test)):
    test_X.append(X_test[i].flatten())

# running bayesian classifier on training data
y_train_pred = gnb.fit(train_X, train_y).predict(train_X)

print("TRAINING DATA - Num. mislabeled points: %d / %d" %
      ((train_y != y_train_pred).sum(), len(train_X)))

# running bayesian classifier on test data
y_test_pred = gnb.fit(train_X, train_y).predict(test_X)

print("TEST DATA - Num. mislabeled points: %d / %d" %
      ((test_y != y_test_pred).sum(), len(test_X)))
