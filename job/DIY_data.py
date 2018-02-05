#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
DIY_data.py
'''

import numpy as np
import pandas as pd
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

def data_from_hft():
    hft_data = pd.read_csv('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/good_data/HFT_XY_unselected.csv')
    names = ["X%s"% i for i in range(200,333)]+['Unnamed: 0', 'realY','predictY']
    # Standardized features
    x_ori = hft_data.drop(names, axis = 1)
    scaler = StandardScaler().fit(x_ori)
    X = scaler.transform(x_ori)
    X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)
    Y = hft_data["realY"]
    X = X[X.index > 150000]
    Y = Y[Y.index > 150000]
    
    return X,Y

def data_from_hft_pca():
    hft_pca = pd.read_csv('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/pca_data/HFT_PCA/hft_pca.csv')
    names = ['realY']
    x_ori = hft_pca.drop(names, axis = 1)
    scaler = StandardScaler().fit(x_ori)
    X = scaler.transform(x_ori)
    X = pd.DataFrame(X, index = x_ori.index, columns = x_ori.columns)
    Y = hft_pca["realY"]
    
    return X,Y

def data_from_input():
    X_input, Y_input = data_from_hft()
 
    return X_input,Y_input
    

    




