#!/usr/bin/python
# -*- coding: utf-8 -*-

# K_line_graph.py

import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from matplotlib.dates import MonthLocator,MONTHLY
import datetime as dt
import pylab

daylinefilespath = '../hs300'
stock_b_code = '000001' #平安银行
MA5 = 5
MA10 = 10
startdate = dt.date(2015, 1, 5)
enddate = dt.date(2017, 1, 5)

def readstkData(rootpath, stockcode):

	filename = rootpath+ '/' + stockcode + '.csv' 
	returndata = pd.read_csv(filename)
	
	return returndata


def main():
	days = readstkData(daylinefilespath, stock_b_code)
	# convert the datetime64 column in the dataframe to 'float days'
	days['date'] = pd.to_datetime(days['date'])  
	days['date'] = mdates.date2num(days['date'].astype(dt.date))
	#time_format = '%Y-%m-%d'
	#days['date']=[dt.datetime.strptime(i, time_format) for i in days['date']]

	Av1 = days['ma5']
	Av2 = days['ma10']
	#quotes = np.array(days) 
	quotes = zip(days['date'], days['open'], days['high'], days['low'], days['close'])
	fig = plt.figure(facecolor='#07000d',figsize=(15,10))
	#fig = plt.figure()
	ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
	candlestick_ohlc(ax1, quotes, width=.6, colorup='#ff1717', colordown='#53c156')
	Label1 = str(MA5)+' SMA'
	Label2 = str(MA10)+' SMA'
    
	ax1.plot(days.date,Av1,'#e1edf9',label=Label1, linewidth=1.5)
	ax1.plot(days.date,Av2,'#4ee6fd',label=Label2, linewidth=1.5)
	ax1.grid(True, color='w')
	ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax1.yaxis.label.set_color("w")
	ax1.spines['bottom'].set_color("#5998ff")
	ax1.spines['top'].set_color("#5998ff")
	ax1.spines['left'].set_color("#5998ff")
	ax1.spines['right'].set_color("#5998ff")
	ax1.tick_params(axis='y', colors='w')
	ax1.tick_params(axis='x', colors='w')

	volumeMin = 0
	ax1v = ax1.twinx()
	ax1v.fill_between(days.date, volumeMin, days.volume, facecolor='#00ffe8', alpha=.4)
	ax1v.axes.yaxis.set_ticklabels([])
	ax1v.grid(False)
	###Edit this to 3, so it's a bit larger
	ax1v.set_ylim(0, 3*days.volume.values.max())
	ax1v.spines['bottom'].set_color("#5998ff")
	ax1v.spines['top'].set_color("#5998ff")
	ax1v.spines['left'].set_color("#5998ff")
	ax1v.spines['right'].set_color("#5998ff")
	ax1v.tick_params(axis='x', colors='w')
	ax1v.tick_params(axis='y', colors='w')

	ax1.set_ylabel('Stock price and Volume')
	ax1.set_title(stock_b_code,color='w')
	plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    
	plt.legend(loc = 'best')
	plt.show()
	
if __name__ == "__main__":
	main()
