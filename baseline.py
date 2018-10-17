import numpy as np
from sklearn import svm

import data_manipulation



''' Returns an SVM Classifier trained on the provided dataset.

    train_x : an n x (RESIZED_SIZE ** 2) numpy array representing
              the image. '''
def create_svm(train_x, train_y, C):
    clf = svm.SVC(C=C)
    clf.fit(train_x, train_y)

    return clf

def predict(clf, val_x, val_y):
    return clf.score(val_x, val_y)
