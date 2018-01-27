#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
DIY_data.py
'''

import numpy as np
import pandas as pd
import datetime as dt
from tools import create_lagged_series,store_data
from sklearn.preprocessing import StandardScaler

def data_from_tushare():
    snpret = pd.read_csv('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/tushare/hs300/20160130__20171231/hs300.csv',
                         index_col="date", encoding='gbk')
    # Standardized features
    x_ori = snpret.drop(['price_change', 'cla_Direction','reg_Direction'], axis = 1)
    scaler = StandardScaler().fit(x_ori)
    X = scaler.transform(x_ori)
    X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)
    Y = snpret["cla_Direction"]
    
    return X,Y

def data_from_input():
    hft_data = pd.read_csv('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/good_data/HFT_XY_unselected.csv')
    names = ["X%s"% i for i in range(200,333)]+['Unnamed: 0', 'realY','predictY']
    # Standardized features
    x_ori = hft_data.drop(names, axis = 1)
    scaler = StandardScaler().fit(x_ori)
    X = scaler.transform(x_ori)
    X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)
    Y = hft_data["realY"]
   
    X_tushare, Y_tushare = data_from_tushare()
 
    #return X,Y
    return X_tushare, Y_tushare

    




