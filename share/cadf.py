#!/usr/bin/python
# -*- coding: utf-8 -*-

# cadf.py  Calculat the CADF test

import datetime
import time
import numpy as np
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts
from sklearn.linear_model import LinearRegression as LR
import itertools 

from tools import get_hs300_symbols, readstkData


def cal_cadf(dl_dir, *symbol):
    num = 30
    data1 = readstkData(dl_dir, symbol[0])
    data2 = readstkData(dl_dir, symbol[1])
    df1 = data1[['date','close']]
    df2 = data2[['date','close']]
    re_df1 = df1.rename(columns={'close':'x_close'})
    re_df2 = df2.rename(columns={'close':'y_close'})
    # solve the problem that 'Found input variables with inconsistent numbers of samples'
    result = pd.merge(re_df1, re_df2, how='inner', on=['date']) 
    x_close = np.array(result['x_close'])
    y_close = np.array(result['y_close'])
    #solve the problem that 'ValueError: maxlag should be < nobs'
    if len(x_close)<num:
        return 'pass' 
    x_new_close = x_close.reshape(-1,1)
    lr = LR(n_jobs = 10)
    lr.fit(x_new_close, y_close)
    res = y_close - lr.predict(x_new_close)
    cadf = ts.adfuller(res)
    return cadf

def main():
    # Make sure you've created this 
    # relative directory beforehand
    dl_dir = '../hs300'
    starttime = time.time()
    codes, symbol_names = get_hs300_symbols()
    symbol_codes = list(itertools.combinations(codes,2))
    #symbol_codes = list(itertools.permutations(codes,2))
    f = open("../doc/hs300_cointegration.dat",'w')
    for c in symbol_codes[0:100]:
        cadf = cal_cadf(dl_dir, *c)
        if cadf != 'pass':
            if cadf[0] < cadf[4]['5%']:
                print >>f,  "%s and %s are cointegration"%(c[0],c[1])
    f.close()
    endtime1 = time.time()
    timeall1 = endtime1 - starttime
    print 'running time %s'%timeall1
if __name__ == "__main__":
    main()

