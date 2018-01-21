#!/usr/bin/python
# -*- coding: utf-8 -*-

# model_tpot.py

from tpot import TPOTClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from tool import create_lagged_series
from sklearn.preprocessing import StandardScaler
import datetime as dt

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

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)
    tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))    

