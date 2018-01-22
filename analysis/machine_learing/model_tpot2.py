#!/usr/bin/python
# -*- coding: utf-8 -*-

# model_tpot.py

from tpot import TPOTRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    hft_data = pd.read_csv('../../good_data/HFT_XY_unselected.csv')
    names = ["X%s"% i for i in range(33,333)]+['Unnamed: 0', 'realY','predictY']
    x_ori = hft_data.drop(names, axis = 1)
    scaler = StandardScaler().fit(x_ori)
    X = scaler.transform(x_ori)
    X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)
    Y = hft_data["realY"]
    

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25)
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))

