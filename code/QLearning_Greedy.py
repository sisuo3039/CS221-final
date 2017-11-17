import pandas_datareader as pdr
from datetime import datetime
import sys
import random
from random import randint
from copy import deepcopy

sap = pdr.get_data_yahoo(symbols='SAP', start=datetime(2015, 1, 1), end=datetime(2017, 10, 26))

# extracts stock data
def data_extract(stock):
	result = []
	date = sap['Open'].keys()

	amount_held = 0

	for i in range(len(stock)):
		result.append((stock['Open'][i], date[i], "Open"))
		result.append((stock['Adj Close'][i], date[i], "Close"))

	return result

# calculates dot product
def dot_product(feature, weight):
	sumNum = 0
	for i in range(len(feature)):
		sumNum += feature[i]*weight[i]

	return sumNum

# append action onto feature
def feature_action_append(feature, action):
	new_feature = deepcopy(feature)
	if action == "buy":
		new_feature += [1,0,0]
	elif action == "sell":
		new_feature += [0,1,0]
	else:
		new_feature += [0,0,1]

	return new_feature

# return (Optimal utility for the current state)
def get_V_Opt(weight, feature, actions):
	opt = 0
	opt_action = None

	for action in actions:
		newFeature = feature_action_append(feature, action)
		dp = dot_product(newFeature,weight) 
		if dp > opt:
			opt = dp
			opt_action = action

	return (opt, action)

# return the current reward for current action and state
def get_reward(action, purchased_price, curr_price):
	if action == "stay" or action=="buy":
		return 0
	elif action == "sell":
		return curr_price - purchased_price

# return possible actions given state
# when we are holding stocks, then action is just "sell" and "stay"
# "buy" and "stay" otherwise
def get_action(amount_held):
	if amount_held == 1:
		return ["sell", "stay"]
	else:
		return ["stay", "buy"]

# input: 
# data - data from stock using data_extract
# weight - weight
# boolean randomPolicy - if true then do random else follow exactly policy pi from weight
# output:
# total profit

def Q_learn(data, weight, randomPolicy=True):
	# profit
	profit = 0

	# hyperparam
	eta = 0.0001
	gamma = 1
	epsilon = 0.2

	#states
	amount_held = 0
	purchased_price = 0
	curr_price = data[1][0]
	prev_price = data[0][0]

	for i in range(2, len(data)-1):
		# get possible actions using current feature
		poss_action = get_action(amount_held)

		# current feature minus action
		feature = [amount_held, curr_price, prev_price, purchased_price]

		# select an action based on policy
		if randomPolicy:
			action = random.choice(poss_action)
		else:
			# let's use epsilon greedy
			if randint(0,9) < epsilon*10:
				action = random.choice(poss_action)
			else:
				# follow policy
				(value, action) = get_V_Opt(weight, feature, poss_action)

		#append action to feature
		feature_added_action = feature_action_append(feature,action)

		# calculate the estimated Q_opt
		Q_opt = dot_product(feature_added_action, weight)

		#reward for the current action
		reward = get_reward(action, purchased_price, curr_price)

		#total profit
		profit += reward

		#update states
		if action == "buy":
			purchased_price = curr_price
			amount_held = 1
		elif action == "sell":
			purchased_price = 0
			amount_held = 0

 		prev_price = curr_price
		curr_price = data[i+1][0]
		new_feature = [amount_held, curr_price, prev_price, purchased_price]

		# Q learning
		amount = (Q_opt-(reward+gamma*get_V_Opt(weight, new_feature, poss_action)[0]))
		feature_added_action = [eta* amount * feature for feature in feature_added_action]
		# updating weight
		for i in range(len(weight)):
			weight[i] = weight[i]-feature_added_action[i] 

	# if stock was never sold in the end, sell it
	if amount_held == 1:
		profit += curr_price - purchased_price
	return profit

weight = [0]*7
data = data_extract(sap)


'''average_profit = 0
for i in range(100):
	profit = Q_learn(data, weight)
	average_profit += profit

print "random policy average", average_profit/100
	
average_profit = 0
for i in range(100):
	profit = Q_learn(data, weight, False)
	average_profit += profit

print "Epsilon greedy policy average", average_profit/100
'''

#print Q_learn(data, weight)
#print Q_learn(data, weight, False)
print weight





# O(N)
# compares today's price with tomorrow's price
def oracle_linear_search(data):
	optimal = []
	profit = 0
	held = False
	date = -1

	boughtProfit = 0

	for i in range(len(data)-1):
		# if price tomorrow is greater than today, what to buy
		if data[i+1][0] > data[i][0]:
			if not held:
				held = True
				date = (data[i][1], data[i][2])
				boughtProfit = data[i][0]
		# if price falls, what to sell
		else:
			if held:
				held = False
				optimal.append("bought at: " + str(date) + " sold at: " + str((data[i][1], data[i][2])))
				profit += data[i][0] - boughtProfit
				cash = shares*data[i][0]
				boughtProfit = 0

	return (optimal, profit)


# greedy algorithm:
# if bought: 
#			sell when price rises
# else:
#			buy when price falls
def greedy(data):
	profit = 0
	boughtPrice = 0
	cash = 1000
	shares = 0

	for i in range(1,len(data)):
		# data[i] in format (price, date, Open/close)
		currPrice = data[i][0]
		if boughtPrice != 0:
			if data[i-1][0] < currPrice:
				reward = currPrice - boughtPrice
				profit += reward
				cash = shares*currPrice
				shares = 0
				boughtPrice = 0
		else: # not holding stock
			if data[i-1][0] > currPrice:
				boughtPrice = currPrice
				shares = cash/currPrice
				cash = 0

	if shares != 0:
		cash = shares*currPrice
	return cash

profit = greedy(data)
print "Greedy yields profit", profit