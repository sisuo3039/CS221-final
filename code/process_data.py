import pandas_datareader as pdr
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.dates as dates
import tensorflow as tf
import actions

# predict next 10 days' stock price based on last ten days' stock price
model_size = 10


## load data
sap_train = pdr.get_data_yahoo(symbols='SAP', start=datetime(2015, 1, 1), end=datetime(2017, 1, 1))
traindata = []
for i in range(len(sap_train)):
    traindata.append(sap_train['Open'][i])
    traindata.append(sap_train['Close'][i])

# print ("total length", len(traindata))
sap_eval = pdr.get_data_yahoo(symbols='SAP', start=datetime(2017, 1, 1), end=datetime(2017, 10, 1))
evaldata = []
for i in range(len(sap_eval)):
    evaldata.append(sap_eval['Open'][i])
    evaldata.append(sap_eval['Close'][i])


## process the data into training data x and label data y
def process_data(data):
    x = []
    y = []
    z = []
    # normalize the data by dividing last day's price
    for i in range(len(data) - model_size - 1):
        y.append(data[i+10]/data[i+9])
        x.append(normalize(data[i:i + 10]))
        z.append(data[i + 10])
    return x, y, z

def process_data(data, model_size):
    x = []
    y = []
    z = []
    # normalize the data by dividing last day's price
    for i in range(len(data) - model_size - 1):
        y.append(data[i+model_size]/data[i+model_size-1])
        x.append(normalize(data[i:i + model_size]))
        z.append(data[i + model_size])
    return x, y, z


def normalize(data_line):
    result = [0.0] * len(data_line)
    result[0] = 1.0
    for i in range(1, len(data_line)):
        result[i] = data_line[i]/data_line[i-1]
    return result

def get_train_data():
    x_train, y_train, train_real_value = process_data(traindata, model_size=model_size)
    return (x_train, y_train, train_real_value)
def get_eval_data():
    x_eval, y_eval, eval_real_value = process_data(evaldata, model_size=model_size)
    return (x_eval, y_eval, eval_real_value)
def get_whole_eval_data():
    return evaldata
def get_whole_train_data():
    return traindata

## get the stock's data for prediction (10 data points before the specified date)
# company: Company's stock name as a string, i.e. 'SAP'
# year, month, day: the stock price of interest
def get_data(company, start_year, start_month, start_day, end_year, end_month, end_day, model_size):
    startdate = datetime(start_year, start_month, start_day) - timedelta(days=model_size/2+1)
    enddate = datetime(end_year, end_month, end_day) - timedelta(days=1)
    stockdata = pdr.get_data_yahoo(symbols=company, start=startdate, end=enddate)
    data = []
    for i in range(len(stockdata)):
        data.append(stockdata['Open'][i])
        data.append(stockdata['Close'][i])
    x, y, z = process_data(data, model_size)
    return x, y, z

sap = pdr.get_data_yahoo(symbols='SAP', start=datetime(2015, 1, 1), end=datetime(2017, 1, 1))

# extracts stock data
def data_extract(stock):
    result = []
    date = sap['Open'].keys()
    print "stock dates", len(stock['High'])
    for i in range(len(stock)):
        # result.append(date)
        result.append((dates.date2num(date[i]), stock['Open'][i], stock['High'][i], stock['Low'][i], stock['Close'][i]))
        # result.append((stock['Adj Close'][i]))
    return result

## single datapoint for each day
def get_train_data_2():
    return data_extract(sap)


def judgeHighLow(trueData, predictedData):
    n = len(trueData)
    rightCalls = 0
    rc_plot = [0] * (n)
    for i in xrange(n):
        if (trueData[i] - 1) * (predictedData[i] - 1) >= 0:
            rightCalls += 1
            rc_plot[i] = 1
    # for i in xrange(n-1):
    #     if (trueData[i+1] - trueData[i]) * (predictedData[i+1] - trueData[i]) >= 0:
    #         rightCalls += 1
    #         rc_plot[i+1] = 1
    # plt.plot(range(n-1), rc_plot)
    # plt.show()
    return rightCalls * 1.0 / (n-1), rc_plot


## return the index function (Y value is 1 if the stock price increases, -1 otherwise)
def index_function(Y):
    print "Y is ", Y
    test =  Y[:-1]
    Y_baseline = tf.concat([[0.0], Y[:-1]], axis=0)
    print "shapes", Y_baseline
    return tf.less(Y_baseline, Y), Y_baseline

# x, y = process_data(evaldata)
# print "test x", x
# print "test y", y


def plotPrices(true_data, predict_data, train_Y):
    # Graphic display
    precision, precision_data = judgeHighLow(train_Y, predict_data)
    print ("the precision to make the right call for training dataset is", precision)
    time = range(len(true_data))
    plt.plot(time, true_data, 'b-', label='True price')
    # time_correct = [i for i in time if precision_data[i] == 1]
    # predict_correct = [predict_data[i]*true_data[i]/train_Y[i] for i in time if precision_data[i] == 1]
    # time_incorrect = [i for i in time if precision_data[i] != 1]
    # predict_incorrect = [predict_data[i]*true_data[i]/train_Y[i] for i in time if precision_data[i] != 1]
    predict_true = [predict_data[i]*true_data[i]/train_Y[i] for i in time]
    plt.plot(time, predict_true, 'r--', label='Predicted price')
    # plt.plot(time_correct, predict_correct, 'r--', label='correct prediction')
    # plt.plot(time_incorrect, predict_incorrect, 'g^', label='incorrect prediction')
    plt.legend()
    plt.show()
    print ("the profit of true value is", actions.calculate_profit(true_data, true_data))
    print ("the profit of predicted value is", actions.calculate_profit(true_data, predict_true))

    #this name tag need to be updated manually
    thefile = open('lstm_model_prediction_data.txt', 'w')
    for item in predict_true:
        thefile.write("%s\n" % item[0])
    thefile.close()
    return
