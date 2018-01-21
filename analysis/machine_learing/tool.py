#!/usr/bin/python
# -*- coding: utf-8 -*-

# tool.py

from __future__ import print_function

import pandas as pd
import numpy as np
import tushare as ts
import datetime as dt
from feature import SMA,EWMA,BBANDS,CCI,EVM,ForceIndex,ROC

def get_hs300_symbols():
    """
    get a list of symbols of hs300
    from tushare.
    """
    symbol = ts.get_hs300s()

    return symbol['code'], symbol['name']

def readstkData(rootpath, stockcode):

    filename = rootpath+ '/' + stockcode + '.csv'
    returndata = pd.read_csv(filename)

    return returndata

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
    tsret["price_change"] = ts_hs300['price_change'].shift(-1)
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["cla_Direction"] = np.sign(tsret["price_change"])
    tsret["reg_Direction"] = ts_hs300['close'].shift(-1)
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

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)



