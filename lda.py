# PROBLEM THREE: FISHER LDA CLASSIFIER - TWO CATEGORIES

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.datasets import mnist

# loading training and test data
(X_train, train_y), (X_test, test_y) = mnist.load_data()

# flattening 28 x 28 element 2d array into 1d array to match lda param requirements
train_X = []
test_X = []

for i in range(len(X_train)):
    train_X.append(X_train[i].flatten())

for i in range(len(X_test)):
    test_X.append(X_test[i].flatten())

# storing training and test data by class to allow for two-category classifications
training_by_class = {}
test_by_class = {}

for i in range(len(train_y)):
    cat = train_y[i]
    if cat not in training_by_class:
        training_by_class[cat] = []
    training_by_class[cat].append(train_X[i])

for i in range(len(test_y)):
    cat = test_y[i]
    if cat not in test_by_class:
        test_by_class[cat] = []
    test_by_class[cat].append(test_X[i])

# initializing LDA and two-category comparisons
lda = LinearDiscriminantAnalysis()
two_cat = [(0, 9), (0, 8), (1, 7)]
print('FISHER LDA TWO-CATEGORY CLASSIFIER')

# performing LDA classifications on each pair of two-category comparisons
for i, j in two_cat:
    print("Digit: %d vs. digit: %d" % (i, j))
    # building training and test data for two given categories i and j
    new_train_X = []
    new_train_y = []
    new_test_X = []
    new_test_y = []
    train_i = training_by_class[i]
    train_j = training_by_class[j]
    test_i = test_by_class[i]
    test_j = test_by_class[j]
    # iterating through data of each class, putting it together to create
    # new training and test data
    for a in range(len(train_i)):
        new_train_X.append(train_i[a])
        new_train_y.append(i)
    for a in range(len(train_j)):
        new_train_X.append(train_j[a])
        new_train_y.append(j)
    for a in range(len(test_i)):
        new_test_X.append(test_i[a])
        new_test_y.append(i)
    for a in range(len(test_j)):
        new_test_X.append(test_j[a])
        new_test_y.append(j)
    # running two category lda classifier on new training data
    y_train_pred = lda.fit(new_train_X, new_train_y).predict(new_train_X)
    print("TRAINING DATA - Num. mislabeled points: %d / %d" %
          ((new_train_y != y_train_pred).sum(), len(new_train_X)))
    # running two category lda classifier on new test data
    y_test_pred = lda.fit(new_train_X, new_train_y).predict(new_test_X)
    print("TEST DATA - Num. mislabeled points: %d / %d" %
          ((new_test_y != y_test_pred).sum(), len(new_test_X)))
