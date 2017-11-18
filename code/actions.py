import pandas_datareader as pdr
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import process_data as pd


# predict next 10 days' stock price based on last ten days' stock price
model_size = 10


## load data
sap_train = pdr.get_data_yahoo(symbols='SAP', start=datetime(2015, 1, 1), end=datetime(2017, 1, 1))
# store Adj Close data into puredata list
traindata = []
traindata.append(sap_train['Open'])
traindata.append(sap_train['Adj Close'])

sap_eval = pdr.get_data_yahoo(symbols='SAP', start=datetime(2017, 1, 1), end=datetime(2017, 10, 1))
# store Adj Close data into puredata list
evaldata = sap_eval['Adj Close']

# ## starting money amount and stock amount
# cash = 1000.
# stock = 0.
# transaction_cost = 0
# ## baseline randomly buy or sell
# for i in range(len(traindata)):
#     # buy
#     if random.randint(1,3) == 1 and cash > 0:
#         stock = (cash - transaction_cost) / traindata[i]
#         cash = 0
#     # sell
#     elif random.randint(1,3) == 2 and stock > 0:
#         cash += stock * traindata[i] - transaction_cost
#         stock = 0
#     accountvalue = cash + stock * traindata[i] - transaction_cost
# print "the total money after 2 years' random trading is", accountvalue


def calculate_profit(truth, data):
    cash = 1000.
    stock = 0.
    transaction_cost = 0.
    print ("total days", len(data))
    ## rational
    for i in range(len(data)):
        currPrice = truth[i][0]
        future_prices = data[i]
        future_high = max(future_prices)
        high_day = np.argmax(future_prices)
        future_low = min(future_prices[:high_day + 1])

        # buy
        # if future_high - transaction_cost > currPrice and cash > 0 and currPrice <= future_low:
        #     stock = (cash - transaction_cost) / currPrice
        #     cash = 0
        # # sell
        # elif currPrice >= future_high and stock > 0:
        #     cash = stock * currPrice - transaction_cost
        #     stock = 0
        # # print ("curprice and future price", currPrice, future_high)

        ## greedy method based on next day's price
        # buy
        if data[i][1] >= currPrice and cash > 0:
            stock = (cash - transaction_cost) / currPrice
            cash = 0
        # sell
        elif data[i][1] < currPrice and stock > 0:
            cash = stock * currPrice - transaction_cost
            stock = 0
        # print ("curprice and future price", currPrice, future_high)


        # print ("the money", stock, cash, data[i][1], currPrice)
    if stock != 0:
        cash = stock * currPrice
    return cash


x_train, y_train = pd.get_train_data()
train_X = np.asarray(x_train)
train_Y = np.asarray(y_train)
x_eval, y_eval = pd.get_eval_data()
eval_X = np.asarray(x_eval)
eval_Y = np.asarray(y_eval)
print ("final profit is", calculate_profit(train_Y, train_Y))
# print ("length of train_Y", len(train_Y))
# print train_Y[0]
# print train_Y[1]
# print train_Y[2]