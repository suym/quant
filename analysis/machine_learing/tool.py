#!/usr/bin/python
# -*- coding: utf-8 -*-

# tool.py

from __future__ import print_function

import pandas as pd
import tushare as ts
import datetime as dt

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


