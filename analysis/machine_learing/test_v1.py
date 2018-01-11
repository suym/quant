#!/usr/bin/python
# -*- coding: utf-8 -*-

# test.py

from __future__ import print_function

import datetime as dt
import numpy as np
import pandas as pd
import tushare as ts
from feature import SMA,EWMA,BBANDS,CCI,EVM,ForceIndex,ROC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC

def create_lagged_series(symbol, start_date, end_date):
    """
    This creates a pandas DataFrame that stores the 
	close value of a stock obtained from Tushare
    """

	# Obtain stock information from Tushare
    ts_hs = ts.get_hist_data(
							symbol, 
							str(start_date-dt.timedelta(days=365)),
							str(end_date)
	)
	# Descending by date
    ts_hs300 = ts_hs.sort_index()

	# Create the new returns DataFrame
    tsret = pd.DataFrame(index=ts_hs300.index)
    tsret["Today"] = ts_hs300['price_change'].shift(-1)
	# Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
	# Create the other colum
    features = ['open','high','close','low','volume']
    for fe in features:
		tsret[fe] = ts_hs300[fe]
    indexs = [("SMA5", SMA(ts_hs300,5)),
              ("SMA10", SMA(ts_hs300,10)),
              ("SMA20", SMA(ts_hs300,20)),
              ("EWMA_20", EWMA(ts_hs300,20)),
              ("BBANDS", BBANDS(ts_hs300,20)),
              ("CCI", CCI(ts_hs300,20)),
              ("EVM", EVM(ts_hs300,20)),
              ("ForceIndex", ForceIndex(ts_hs300,20)),
              ("ROC", ROC(ts_hs300,20)),
				]
    for ind in indexs:
		tsret_1 = ind[1]
		tsret=tsret.join(tsret_1)	

    tsret = tsret[tsret.index >= str(start_date)]

	# To remove a null value
    tsret=tsret.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    return tsret


if __name__ == "__main__":
    # Create a lagged series of the hs300 stock market index
    snpret = create_lagged_series(
        "hs300",
		dt.datetime(2016, 1, 30),
        dt.datetime(2017, 12, 31)
    )
	
    # Use the prior two days of returns as predictor 
    # values, with direction as the response
    X = snpret[["SMA5","SMA10","SMA20","EWMA_20","Upper BollingerBand","Upper BollingerBand",
				"Lower BollingerBand","CCI","EVM","ForceIndex","Rate of Change"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2017.
    start_test = str(dt.datetime(2017,1,1))

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

