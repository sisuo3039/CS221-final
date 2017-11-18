import pandas_datareader as pdr
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import random
import numpy as np

# predict next 10 days' stock price based on last ten days' stock price
model_size = 10


## load data
sap_train = pdr.get_data_yahoo(symbols='SAP', start=datetime(2015, 1, 1), end=datetime(2017, 1, 1))
# store Adj Close data into puredata list
# traindata = sap_train['Adj Close']
traindata = np.concatenate((sap_train['Open'], sap_train['Adj Close']))

print ("total length", len(traindata))
sap_eval = pdr.get_data_yahoo(symbols='SAP', start=datetime(2017, 1, 1), end=datetime(2017, 10, 1))
# store Adj Close data into puredata list
# evaldata = sap_eval['Adj Close']
traindata = np.concatenate((sap_eval['Open'], sap_eval['Adj Close']))

# ## starting money amount and stock amount
# cash = 1000.
# stock = 0.
# ## baseline randomly buy or sell
# for i in range(len(traindata)):
#     # buy
#     if random.randint(1,3) == 1 and cash > 0:
#         stock = cash / traindata[i]
#         cash = 0
#     # sell
#     elif random.randint(1,3) == 2 and stock > 0:
#         cash += stock * traindata[i]
#         stock = 0
#     accountvalue = cash + stock * traindata[i]
# print "the total money after 2 years' random trading is", accountvalue


## process the data into training data x and label data y
def process_data(data):
    x = []
    y = []
    for i in range(len(data) - 2*model_size):
        x.append(data[i:i+10])
        y.append(data[i+10:i+20])
    return x, y

def get_train_data():
    x_train, y_train = process_data(traindata)
    return (x_train, y_train)
def get_eval_data():
    x_eval, y_eval = process_data(evaldata)
    return (x_eval, y_eval)
def get_whole_eval_data():
    return evaldata
def get_whole_train_data():
    return traindata

# print " x_train, y_train shapes are", len(x_train), len(x_train[0]), len(y_train), len(y_train[0])
# print "x_eval, y_eval shapes are", len(x_eval), len(x_eval[0]), len(y_eval), len(y_eval[0])
# x_eval, y_eval = get_eval_data()
#
# print (len(evaldata))