# -*- coding: utf-8 -*-

# get_hs300_data.py

from __future__ import print_function

import matplotlib.pyplot as plt
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


def download_hs300_from_tushare(code, dl_dir, start_year, end_year):
	"""
	Download stocks of hs300 from tushare and then
	store it to disk in the 'dl_dir' directory. 
	"""
	# Download the data from tushare
	data = ts.get_hist_data(code, start_year, end_year)

	# Store the data to disk
	data.to_csv('%s/%s.csv' % (dl_dir, code))


def download_historical_hs300(dl_dir, start_year, end_year):
	"""
	Downloads hs300 data from tushare
	between a start_year and an end_year.
	"""
	symbol_codes, symbol_names = get_hs300_symbols()
	for c in symbol_codes:
		print("Downloading code: %s" % c)
		download_hs300_from_tushare(c, dl_dir, start_year, end_year)


if __name__ == "__main__":

    # Make sure you've created this 
    # relative directory beforehand
	dl_dir = '../hs300'

	# Create the start and end years
	start_year =  str(dt.date(2015, 1, 5))
	end_year = str(dt.date(2017, 1, 5))

	# Download the hs300 data into the directory
	download_historical_hs300(dl_dir, start_year, end_year)


