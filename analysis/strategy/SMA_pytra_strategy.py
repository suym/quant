#!/usr/bin/python
# coding=utf-8  

from pyalgotrade import strategy   
from pyalgotrade.technical import ma  

# 1.构建一个策略  
class SMA_strategy(strategy.BacktestingStrategy):  
    def __init__(self, feed, instrument, brk):  
        super(SMA_strategy, self).__init__(feed, brk)  
        self.__position = None  
        self.__sma = ma.SMA(feed[instrument].getCloseDataSeries(), 150)  
        self.__instrument = instrument  
        self.getBroker()  
    def onEnterOk(self, position):  
        execInfo = position.getEntryOrder().getExecutionInfo()  
        self.info("BUY at %.2f" % (execInfo.getPrice()))  
  
    def onEnterCanceled(self, position):  
        self.__position = None  
  
    def onExitOk(self, position):  
        execInfo = position.getExitOrder().getExecutionInfo()  
        self.info("SELL at $%.2f" % (execInfo.getPrice()))  
        self.__position = None  
  
    def onExitCanceled(self, position):  
        # If the exit was canceled, re-submit it.  
        self.__position.exitMarket()  
  
    def getSMA(self):  
        return self.__sma  
  
    def onBars(self, bars):# 每一个数据都会抵达这里，就像becktest中的next  
  
        # Wait for enough bars to be available to calculate a SMA.  
        if self.__sma[-1] is None:  
            return  
        #bar.getTyoicalPrice = (bar.getHigh() + bar.getLow() + bar.getClose())/ 3.0  
  
        bar = bars[self.__instrument]  
        # If a position was not opened, check if we should enter a long position.  
        if self.__position is None:  
            if bar.getPrice() > self.__sma[-1]:  
                # 开多头.  
                self.__position = self.enterLong(self.__instrument, 10, True)  
        # 平掉多头头寸.  
        elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():  
            self.__position.exitMarket()  
            
            