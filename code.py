#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd


# In[3]:


#reading data file from local directory
df1=pd.read_csv(r'C:\Users\Admin\.spyder-py3\TECHM.CSV',parse_dates =["Date"], index_col ="Date")
df1


# In[4]:


#considering only close column and storing in df2 variable
df2=df1[['Close']]
df2
plt.plot(df2)


# In[40]:


#impoerting gdp file in python and storing into df3 variable
df3=pd.read_csv(r'C:\Users\Admin\.spyder-py3\GDP.CSV',parse_dates =["Date"], index_col ="Date")
df3


# In[41]:


#joining gdp with close  
df4=df2.join(df3)
df4


# In[42]:


df4.columns


# In[43]:


#changed the gdp column name from value to gdp
df4.columns=['Close', 'GDP']


# In[44]:


df4


# In[45]:


#interpolateing yearly gdp data into daily data 
df4.interpolate(method="linear",inplace=True, limit_direction ='both')
df4


# In[46]:


#importing inflation  file in python and storing into df3 variable
df5=pd.read_csv(r'C:\Users\Admin\.spyder-py3\inflation.CSV',parse_dates =["Date"], index_col ="Date")
df5


# In[47]:


#joining inflation to close and gdp and storing it into df6 variable
df6=df4.join(df5)
df6


# In[48]:


df6.columns


# In[49]:


#changing the inflation column name
df6.columns=['Close', 'GDP', 'inflation']


# In[50]:


df6


# In[51]:


#interpolation
df6.interpolate(method="linear",inplace=True,limit_direction='both')
df6


# In[52]:


#importing consumer confidence(cc)
df7=pd.read_csv(r'C:\Users\Admin\.spyder-py3\cc.CSV',parse_dates =["Date"], index_col ="Date")
df7


# In[53]:


#joining
df8=df6.join(df7)
df8


# In[54]:


#interpolation
df8.interpolate(method="linear",inplace=True, limit_direction ='both')
df8


# In[55]:


#importing unemployment
df9=pd.read_csv(r'C:\Users\Admin\.spyder-py3\unemp.CSV',parse_dates =["Date"], index_col ="Date")
df9


# In[56]:


#joining 
df10=df8.join(df9)
df10


# In[57]:


#interpolation
df10.interpolate(method="linear",inplace=True, limit_direction ='both')
df10
#df10.to_csv("techmfinal.csv")


# In[58]:


# A variable for predicting 'n' days out into the future
forecast_out = 10 
#Create another column (the target ) shifted 'n' units up
df10['Prediction'] = df10[['Close']].shift(-forecast_out)
#print the new data set
print(df10.tail())


# In[59]:


#independent vaeiables 
X = np.array(df10.drop(['Close','Prediction'],1))
X = X[:-forecast_out]
print(X)


# In[60]:


# Convert the dataframe to a numpy array 
y = np.array(df10['Prediction'])
#dependent variable
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)


# In[61]:


#spliting data into test and train 20% and 80%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[62]:


#svm code for creating a model 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)


# In[63]:


#checking the score of the model to know how good it is
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)


# In[64]:


x_forecast = np.array(df10.drop(['Close','Prediction'],1))[-forecast_out:]
print(x_forecast)


# In[65]:


#x_forecast = np.array(df10.drop(['Prediction'],1))[-forecast_out:]
#print(x_forecast)


# In[66]:


svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)


# In[67]:


lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)


# In[68]:


lr_prediction = lr.predict(x_forecast)
print(lr_prediction)


# In[69]:


plt.plot(df10.Close)


# In[71]:


lrconfidence = lr.score(x_test, y_test)
print("LR confidence: ", lrconfidence)


# In[ ]:





# In[ ]:




