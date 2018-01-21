# Load the necessary packages and modules
import pandas as pd

# Simple Moving Average 
def SMA(data, ndays):
    SMA = pd.Series(data['close'].rolling(window =ndays).mean(), name = 'SMA' + str(ndays))
    data = data.join(SMA)
    return SMA

# Exponentially-weighted Moving Average 
def EWMA(data, ndays):
    EMA = pd.Series(data['close'].ewm(ignore_na=False,span = ndays,min_periods = ndays-1,adjust=True).mean(),
    name = 'EWMA_' + str(ndays))
    data = data.join(EMA)
    return EMA

# Compute the Bollinger Bands 
def BBANDS(data, ndays):

    MA = pd.Series(data['close'].rolling(window=ndays,center=False).mean())
    SD = pd.Series(data['close'].rolling(window=ndays,center=False).std())

    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = 'Upper BollingerBand')
    data = data.join(B1)

    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = 'Lower BollingerBand')
    data = data.join(B2)

    return B1,B2

# Commodity Channel Index 
def CCI(data, ndays):
    TP = (data['high'] + data['low'] + data['close']) / 3
    CCI = pd.Series((TP - TP.rolling(window=ndays,center=False).mean()) / (0.015 * TP.rolling(window= ndays,center=False).std()),
    name = 'CCI')
    data = data.join(CCI)
    return CCI

# Ease of Movement 
def EVM(data, ndays):
    dm = ((data['high'] + data['low'])/2) - ((data['high'].shift(1) + data['low'].shift(1))/2)
    br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
    EVM = dm / br
    EVM_MA = pd.Series(EVM.rolling(window = ndays, center = False).mean(), name = 'EVM')
    data = data.join(EVM_MA)
    return EVM_MA

# Force Index 
def ForceIndex(data, ndays):
    FI = pd.Series(data['close'].diff(ndays) * data['volume'], name = 'ForceIndex')
    data = data.join(FI)
    return FI

# Rate of Change (ROC)
def ROC(data,ndays):
    N = data['close'].diff(ndays)
    D = data['close'].shift(ndays)
    ROC = pd.Series(N/D,name='Rate of Change')
    data = data.join(ROC)
    return ROC

