#!/bin/bash

Work_Dir=/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/quant
#Name=buy_and_hold
#Name=golden_cross
#Name=macd
#Name=rsi
#Name=turtle
Name=pair_trading
Start_date='2015-01-01'
End_date='2015-06-01'
Money=100000

Strategy=${Work_Dir}/rqalpha/strategy/${Name}.py
Data=${Work_Dir}/rqalpha/bundle/
Result=${Work_Dir}/rqalpha/job/${Name}_result.pkl

rqalpha run -f ${Strategy} -d ${Data} -s ${Start_date} -e ${End_date} -o ${Result} --account stock ${Money} --benchmark 000300.XSHG 



