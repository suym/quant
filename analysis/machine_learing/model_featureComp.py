#!/usr/bin/python
# -*- coding: utf-8 -*-

# model_featureComp.py

import numpy as np
import pandas as pd
import datetime as dt
from tool import create_lagged_series, features_com
from sklearn.preprocessing import StandardScaler


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
    Y = snpret["cla_Direction"]

    ranks=features_com(X,Y)
    print ranks

    
