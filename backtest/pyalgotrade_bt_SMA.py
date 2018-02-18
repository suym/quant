#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
pyalgotrade_bt_SMA.py
'''

from pyalgotrade import plotter  
from pyalgotrade.stratanalyzer import sharpe, drawdown ,trades
from pyalgotrade.bar import Frequency  
from pyalgotrade.barfeed.csvfeed import GenericBarFeed 
from pyalgotrade import broker
from SMA_pytra_strategy import SMA_strategy

if __name__ == "__main__":
    feed = GenericBarFeed(Frequency.DAY, None, None)
    path='../notebook'
    filename='fd.csv'
    filepath = path+'/'+filename
    feed.addBarsFromCSV("fd", filepath)  
  
  
    broker_commission = broker.backtesting.TradePercentage(0.0001)  
    
    fill_stra = broker.fillstrategy.DefaultStrategy(volumeLimit=0.1)  
    sli_stra = broker.slippage.NoSlippage()  
    fill_stra.setSlippageModel(sli_stra)  
     
    brk = broker.backtesting.Broker(1000000, feed, broker_commission)  
    brk.setFillStrategy(fill_stra)  
     
  
    myStrategy = SMA_strategy(feed, "fd", brk)   

    sharpe_ratio = sharpe.SharpeRatio()  
    myStrategy.attachAnalyzer(sharpe_ratio)  
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer) 
    trade_situation = trades.Trades()  
    myStrategy.attachAnalyzer(trade_situation) 
    plt = plotter.StrategyPlotter(myStrategy)  
    
    myStrategy.run()  
    myStrategy.info("Final portfolio value: $%.2f" % myStrategy.getResult())  
  
    
    print "sharpe_ratio", sharpe_ratio.getSharpeRatio(0)  
    print "Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100)
    print "total number of trades", trade_situation.getCount()  
    print "commissions for each trade",trade_situation.getCommissionsForAllTrades() 
    
    plt.plot()

    
