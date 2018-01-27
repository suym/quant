#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
store_data_from_tushare.py
'''

import datetime as dt
from tools import create_lagged_series,store_data


if __name__ == "__main__":

        start_date = dt.datetime(2016, 1, 30)
        end_date = dt.datetime(2017, 12, 31)

        store_data(
                        "hs300",
                        start_date,
                        end_date
                    )


