#!/usr/bin/python
# -*- coding: utf-8 -*-

# model_selection.py

from __future__ import print_function

import datetime as dt
import numpy as np
import pandas as pd
from tool import create_lagged_series

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC


if __name__ == "__main__":
    # Create a lagged series of the hs300 stock market index
    snpret = create_lagged_series(
        "hs300",
        dt.datetime(2016, 1, 30),
        dt.datetime(2017, 12, 31)
    )
    # Standardized features
    x_ori = snpret.drop(['price_change', 'cla_Direction','reg_Direction'], axis = 1)
    scaler = StandardScaler().fit(x_ori)
    X = scaler.transform(x_ori)
    X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)	
    y = snpret["cla_Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2017.
    start_test = str(dt.datetime(2017,6,1))

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()),
              ("LDA", LDA()),
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
                C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
                n_estimators=1000, criterion='gini',
                max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features='auto',
                bootstrap=True, oob_score=False, n_jobs=10,
                random_state=None, verbose=0)
              )]

    # Iterate through the models
    for m in models:

        # Train each of the models on the training set
        m[1].fit(X_train, y_train)
        # Make an array of predictions on the test set
        pred = m[1].predict(X_test)

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        print("%s\n" % confusion_matrix(pred, y_test))

