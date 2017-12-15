#!/usr/bin/python

################ Bollinger Bands #############################

# Load the necessary packages and modules
import pandas as pd
import pandas_datareader.data as web

# Compute the Bollinger Bands 
def BBANDS(data, ndays):

	MA = pd.Series(data['Close'].rolling(window=ndays,center=False).mean()) 
	SD = pd.Series(data['Close'].rolling(window=ndays,center=False).std())

	b1 = MA + (2 * SD)
	B1 = pd.Series(b1, name = 'Upper BollingerBand') 
	data = data.join(B1) 

	b2 = MA - (2 * SD)
	B2 = pd.Series(b2, name = 'Lower BollingerBand') 
	data = data.join(B2) 
 
	return data
 
# Retrieve the Nifty data from Yahoo finance:
data = web.DataReader('^NSEI',data_source='yahoo',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data)

# Compute the Bollinger Bands for NIFTY using the 50-day Moving average
n = 5
NIFTY_BBANDS = BBANDS(data, n)
print(NIFTY_BBANDS[0:10])
