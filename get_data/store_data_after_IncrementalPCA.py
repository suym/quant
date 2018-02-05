#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
store_data_after_IncrementalPCA.py
'''

import sys
sys.path.append('../job')

import datetime as dt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from DIY_data import data_from_input
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    X,Y = data_from_input()
    path = '/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/pca_data/HFT_PCA'
    filename = 'hft_pca'
    
    with open('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant/run/inf_log/reg_log.dat','r') as f:
        lines = f.readlines()
        for line in lines:
            if 'IncrementalPCA' in line:
                ipca_par = line.split('DIY')[1]
                ipca_par = eval(ipca_par)
                
    nums = ipca_par['99%']
    names = ["X%s"% i for i in range(nums)]
    
    pca = IncrementalPCA(n_components=nums)
    pca.fit(X)
    X_r = pca.transform(X)
    X_r = pd.DataFrame(X_r, index = X.index, columns = names)
    X_Y = X_r.join(Y)
    X_Y.to_csv('%s/%s.csv'%(path,filename))
    
    print 'X shape:',X_r.shape
    print 'PCA nums:',nums
    print 'shape of X add Y: ',X_Y.shape

    



