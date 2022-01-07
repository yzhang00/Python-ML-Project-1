# PROBLEM FOUR: PCA PLOTTING AND CLASSIFICATION

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import numpy as np

# loading training and test data
(X_train, train_y), (X_test, test_y) = mnist.load_data()

# flattening 28 x 28 element 2d array into 1d array to match gnb param requirements
train_X = []
test_X = []

for i in range(len(X_train)):
    train_X.append(X_train[i].flatten())

for i in range(len(X_test)):
    test_X.append(X_test[i].flatten())

# begin pca plotting for n = 2, 3
print('PCA PLOTTING AND CLASSIFICATION')
for i in [2, 3]:
    # initializing pca
    pca = PCA(i)
    pca.fit(train_X)
    new_train_X = pca.transform(train_X)
    # plotting training data with pca reduced dimensions
    pyplot.figure()
    pyplot.scatter(*new_train_X.T, c=train_y)
    pyplot.title("Training Data Reduced to %d Dimensions" % i)
    pyplot.show()


# begin pca classification for n = 5, 10, 20, 50, 100
# initializing arrays needed for plotting error rate curves
# for bayes and nearest neighbor training and test data
pca_class_dim = [5, 10, 20, 50, 100]
bayes_train_error = []
bayes_test_error = []
neighbor_train_error = []
neighbor_test_error = []
gnb = GaussianNB()
neigh = KNeighborsClassifier(n_neighbors=5)

for i in pca_class_dim:
    print("PCA Classifications for n reduced to %d" % (i))
    pca = PCA(i)
    # generating reduced dimension training and test data
    pca.fit(train_X)
    new_train_X = pca.transform(train_X)
    pca.fit(test_X)
    new_test_X = pca.transform(test_X)
    # running bayesian classifer on reduced training data
    y_train_pred = gnb.fit(new_train_X, train_y).predict(new_train_X)
    print("BAYESIAN TRAINING DATA - Num. mislabeled points: %d / %d" %
          ((train_y != y_train_pred).sum(), len(new_train_X)))
    bayes_train_error.append(
        (train_y != y_train_pred).sum() / len(new_train_X))
    # running bayesian classifier on reduced test data
    y_test_pred = gnb.fit(new_train_X, train_y).predict(new_test_X)
    print("BAYESIAN TEST DATA - Num. mislabeled points: %d / %d" %
          ((test_y != y_test_pred).sum(), len(new_test_X)))
    bayes_test_error.append((test_y != y_test_pred).sum() / len(new_test_X))
    # running nearest neighbor classifier on reduced training data
    y_train_pred = neigh.fit(new_train_X, train_y).predict(new_train_X)
    print("NEAREST NEIGHBOR TRAINING DATA - Num. mislabeled points: %d / %d" %
          ((train_y != y_train_pred).sum(), len(new_train_X)))
    neighbor_train_error.append((train_y != y_train_pred).sum() / len(train_X))
    # running nearest neighbor classifier on reduced test data
    y_test_pred = neigh.fit(new_train_X, train_y).predict(new_test_X)
    print("NEAREST NEIGHBOR TEST DATA - Num. mislabeled points: %d / %d" %
          ((test_y != y_test_pred).sum(), len(new_test_X)))
    neighbor_test_error.append((test_y != y_test_pred).sum() / len(test_X))

# plotting training and test error curves for bayes and nearest neighbor classifiers
pyplot.figure()
pyplot.plot(pca_class_dim, bayes_train_error,
            label="bayes training data error rate")
pyplot.plot(pca_class_dim, bayes_test_error,
            label="bayes test data error rate")
pyplot.xlabel("PCA Reduced Dimensions")
pyplot.ylabel("Error Rate")
pyplot.title(
    "PCA Reduced Dimensions vs. Bayes Classifier Error Rate for Training and Test Data")
pyplot.legend()
pyplot.figure()
pyplot.plot(pca_class_dim, neighbor_train_error,
            label="nearest neighbor training data error rate")
pyplot.plot(pca_class_dim, neighbor_test_error,
            label="nearest neighbor test data error rate")
pyplot.xlabel("PCA Reduced Dimensions")
pyplot.ylabel("Error Rate")
pyplot.title(
    "PCA Reduced Dimensions vs. Nearest Neighbor Classifier Error Rate for Training and Test Data")
pyplot.legend()
pyplot.show()
