#!/usr/bin/python
# -*- coding: utf-8 -*-

# model2.py

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from tool import create_lagged_series
import datetime as dt

if __name__ == "__main__":
    # Create a lagged series of the hs300 stock market index
	snpret = create_lagged_series(
        "hs300",
        dt.datetime(2016, 1, 30),
        dt.datetime(2017, 12, 31)
    )
	X = snpret[["SMA5","SMA10","SMA20","EWMA_20","Upper BollingerBand","Upper BollingerBand",
                "Lower BollingerBand","CCI","EVM","ForceIndex","Rate of Change"]]
	y = snpret["cla_Direction"]

	X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)
	tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
	tpot.fit(X_train, y_train)
	print(tpot.score(X_test, y_test))







