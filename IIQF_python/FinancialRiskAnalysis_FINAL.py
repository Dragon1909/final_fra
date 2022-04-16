#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import datetime
from scipy.stats import norm
from arch import arch_model
from pandas_datareader import data
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')


# In[18]:


axp = pd.read_csv(r"C:\Users\LENOVO\Downloads\AXP.csv",parse_dates = ['Date'])
spy = pd.read_csv(r"C:\Users\LENOVO\Downloads\SPY.csv", parse_dates = ['Date'])


# In[19]:


data = pd.concat([axp['Close'], spy['Close'], axp['Date']], axis = 1)


# In[20]:


data.columns = ['AXP_Close', 'SPY_Close', 'Date']
data.head()


# In[21]:


#Filtering Dates as of December 31, 2021
data = data[data['Date'] < '2022-01-01']
data


# In[22]:


#Solution 2.1

print('AXP')
print('\n')
AXP_garch=arch_model(np.log(data['AXP_Close']/data['AXP_Close'].shift(1)).dropna()*100,p=1,q=1)
AXP_garch_result=AXP_garch.fit(update_freq=10)
print(AXP_garch_result.params)
print(AXP_garch_result.summary())

print('\n')

print('SPY')
print('\n')
SPY_garch=arch_model(np.log(data['SPY_Close']/data['SPY_Close'].shift(1)).dropna()*100,p=1,q=1)
SPY_garch_result=SPY_garch.fit(update_freq=10)
print(SPY_garch_result.params)
print(SPY_garch_result.summary())


# In[23]:


#Solution 2.2 and 2.3

#AXP
variance_AXP_result=np.sqrt(AXP_garch_result.forecast(horizon=1).variance.iloc[-1])
Vl_AXP=np.sqrt((AXP_garch_result.params['omega']/10000)/(1-AXP_garch_result.params['alpha[1]']-                                                         AXP_garch_result.params['beta[1]']))
print("Volatility on 3rd January: ",variance_AXP_result)
print("Long Term Volatility: ",round(Vl_AXP*100,3),"%")

#SPY
variance_SPY_last=np.sqrt(SPY_garch_result.forecast(horizon=1).variance.iloc[-1])
Vl_SPY=np.sqrt((SPY_garch_result.params['omega']/10000)/(1-SPY_garch_result.params['alpha[1]']-                                                         SPY_garch_result.params['beta[1]']))
print("Volatility on 3rd January: ",variance_SPY_last)
print("Long Term Volatility: ",round(Vl_SPY*100,3),"%")


# In[24]:


#Solution 3
#Correlation of returns
np.log(data.drop(['Date'], axis = 1)/data.drop(['Date'], axis = 1).shift(1)).corr()


# In[11]:


#Solution for 4.1

#AXP VaR
axp_data = np.log(data['AXP_Close']/data['AXP_Close'].shift(1))
axp_data = axp_data.dropna()
AXP_VaR_99 = np.percentile(axp_data,1)
print('The daily VaR for AXP is {}'.format(round(AXP_VaR_99*1000000*-1,4)))

#AXP ES
value = 0
count = 0
for i in range(1,axp_data.shape[0]+1):
    if axp_data[i] <= AXP_VaR_99:
        value += axp_data[i]
        count += 1
print('The expected shortfall for AXP is {}'.format(round(value/count*1000000*-1,4)))

print('\n')

#SPY VaR
spy_data = np.log(data['SPY_Close']/data['SPY_Close'].shift(1))
spy_data = spy_data.dropna()
SPY_VaR_99 = np.percentile(spy_data,1)
print('The daily VaR for SPY is {}'.format(round(SPY_VaR_99*1000000*-1,4)))

#SPY ES
value = 0
count = 0
for i in range(1,spy_data.shape[0]+1):
    if spy_data[i] < SPY_VaR_99:
        value += spy_data[i]
        count += 1
print('The expected shortfall for SPY is {}'.format(round(value/count*1000000*-1,4)))

print('\n')

#Portfolio VaR
port_data = axp_data*0.5 + spy_data*0.5
port_VaR_99 = np.percentile(port_data,1)
print('The daily VaR for portfolio is {}'.format(round(port_VaR_99 * 2000000*-1, 4)))

#Portfolio ES
value = 0
count = 0
for i in range(1,port_data.shape[0]+1):
    if port_data[i] < port_VaR_99:
        value += port_data[i]
        count += 1
print('The expected shortfall for portfolio is {}'.format(round(value/count*2000000*-1,4)))


# In[12]:


#Solution for 4.2

#Backtest
ticker=['AXP']
investment=1000000
data_tic = pdr.get_data_yahoo(ticker, start="2019-01-01", end="2021-12-31")['Close']
data_backtest= pdr.get_data_yahoo(ticker, start="2021-03-01", end="2022-02-28")['Close']
no_of_share=investment/data_tic.loc['2021-12-31']['AXP']
change_in_price=[]
for i in range(len(data_backtest)-1):
    change_in_price.append(np.log(data_backtest['AXP'][i+1]/data_backtest['AXP'][i]))

exceed = [ i for i in change_in_price if i < AXP_VaR_99]
print("Only", len(exceed), "time the actual VaR exceeded the calculated VaR")


# In[13]:


#Solution for 5.1

weights = np.array([0.5,0.5])
port_data = port_data.dropna()
mean_data = port_data.mean()
cov_data = np.log(data.drop('Date', axis = 1)/data.drop('Date', axis = 1).shift(1)).cov()

port_return=np.log(data.drop('Date', axis = 1)/data.drop('Date', axis = 1).shift(1)).mean().dot(weights)
port_stdev = np.sqrt(weights.T.dot(cov_data).dot(weights))
conf_level1 = 0.01

print('The portfolio VaR using normal distribution assumption is {}'      .format(round(norm.ppf(conf_level1, port_return, port_stdev)*-1*2000000,4)))


# In[16]:


#Solution for 6

#AXP
investment=1000000
ticker = ['AXP']

data = pdr.get_data_yahoo(ticker, start="2019-01-01", end="2022-01-03")['Close']
Price_ups_1 = data.loc['2021-12-31']['AXP']
Price_ups_2 = data.loc['2022-01-03']['AXP']

No_of_share = investment/Price_ups_1

Change_in_price_ups = Price_ups_2 - Price_ups_1

gain_or_loss = Change_in_price_ups*No_of_share

RAROC_A=(gain_or_loss/(AXP_VaR_99*investment*-1))*100
print("The actual profit and loss on Jan 3rd, 2021 for AXP is:", round(gain_or_loss,4))
print("The Risk-Adjusted Return On Capital (RAROC) for AXP is:",round(RAROC_A,4),"%")

print('\n')

#SPY
investment=1000000
ticker = ['^GSPC']

data = pdr.get_data_yahoo(ticker, start="2019-01-01", end="2022-01-03")['Close']
Price_spy_1 = data.loc['2021-12-31']['^GSPC']
Price_spy_2 = data.loc['2022-01-03']['^GSPC']

No_of_share = investment/Price_spy_1

Change_in_price_spy = Price_spy_2 - Price_spy_1

gain_or_loss = Change_in_price_spy*No_of_share

RAROC_B=(gain_or_loss/(SPY_VaR_99*investment*-1))*100
print("The actual profit and loss on Jan 3rd, 2021 for SPY is: ", round(gain_or_loss,4))
print("The Risk-Adjusted Return On Capital (RAROC) for SPY is: ",round(RAROC_B,4),"%")

print('\n')

investment=1000000 #Investing 1 Million in both SPY and WFC
ticker = ['AXP','^GSPC']

data = pdr.get_data_yahoo(ticker, start="2019-01-01", end="2022-01-03")['Close']
Price_spy_1 = data.loc['2021-12-31']['^GSPC']
Price_spy_2 = data.loc['2022-01-03']['^GSPC']

No_of_share = investment/Price_spy_1
Change_in_price_spy = Price_spy_2 - Price_spy_1

gain_or_loss_spy = Change_in_price_spy*No_of_share

Price_wfc_1 = data.loc['2021-12-31']['AXP']
Price_wfc_2 = data.loc['2022-01-03']['AXP']

No_of_share = investment/Price_wfc_1
Change_in_price_wfc = Price_wfc_2 - Price_wfc_1

gain_or_loss_wfc = Change_in_price_wfc*No_of_share
Total_gain_or_loss = gain_or_loss_spy+gain_or_loss_wfc

RAROC_C=(Total_gain_or_loss/(port_VaR_99*investment*2*-1))*100
print("The actual profit and loss on Jan 3rd, 2021 for Portfolio C is: ", round(Total_gain_or_loss,4))
print("The Risk-Adjusted Return On Capital (RAROC) for Portfolio C is: ",round(RAROC_C,4),"%")


# In[ ]:




