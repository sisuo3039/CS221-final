import pandas_datareader as pdr
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import process_data as pd


# predict next 10 days' stock price based on last ten days' stock price
model_size = 10


def calculate_profit(truth, data):
    cash = 1000.
    shares = 0.
    boughtPrice = 0.
    ## rational
    for i in range(0, len(data)-1):   #len(data)
        # data[i] in format (price, date, Open/close)
        currPrice = truth[i]
        # sell
        if boughtPrice != 0:
            if data[i + 1] < currPrice and shares > 0:
                cash = shares * currPrice
                shares = 0.
                boughtPrice = 0.
        ## buy
        else:  # not holding stock
            if data[i + 1] > currPrice and cash > 0:
                boughtPrice = currPrice
                shares = cash / currPrice
                cash = 0.
        # print ("the money", shares, cash, currPrice)
    if shares != 0:
        cash = shares * currPrice
    return cash

def greedy(data):
    profit = 0
    boughtPrice = 0
    cash = 1000.
    shares = 0.
    cost = 1.
    total_transactions = 0
    for i in range(0, len(data)-1):   #len(data)
        # data[i] in format (price, date, Open/close)
        currPrice = data[i]
        # sell
        if boughtPrice != 0:
            if data[i + 1] < currPrice and shares > 0:
                total_transactions += 1
                cash = shares * currPrice
                shares = 0.
                boughtPrice = 0.
        ## buy
        else:  # not holding stock
            if data[i + 1] > currPrice and cash > 0:
                boughtPrice = currPrice
                shares = cash / currPrice
                cash = 0.
    if shares != 0:
        cash = shares * currPrice
        print ("shares ", shares)
    return cash

def greedy_with_lastday(data):
    profit = 0
    boughtPrice = 0
    cash = 1000.
    shares = 0.
    cost = 1.
    total_transactions = 0
    for i in range(1, len(data)):   #len(data)
        # data[i] in format (price, date, Open/close)
        currPrice = data[i]
        # sell
        if boughtPrice != 0:
            if data[i-1] <= currPrice and shares > 0:
                total_transactions += 1
                cash = shares * currPrice
                shares = 0.
                boughtPrice = 0.
        ## buy
        else:  # not holding stock
            if data[i-1] > currPrice and cash > 0:
                boughtPrice = currPrice
                shares = cash / currPrice
                cash = 0.
    if shares != 0:
        cash = shares * currPrice
    return cash



# data = pd.get_train_data_2()
# open_close_data = []
# for d in data:
#     open_close_data.append(d[1])
#     open_close_data.append(d[4])
#
# print "the raw data is ", len(pd.traindata)
# print "the raw data2 is ", len(open_close_data)
#
# print "greedy method", greedy(open_close_data)
# print "greedy method 2", calculate_profit(open_close_data, open_close_data)
# print "greedy method for old dataset", greedy_with_lastday(open_close_data)

