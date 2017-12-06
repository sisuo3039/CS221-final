


# --- enable this to run on GPU
# import os
#
# os.environ['THEANO_FLAGS'] = "device=cuda*,floatX=float32"

import random, numpy, math,csv,scipy
from SumTree import SumTree

from datetime import datetime ,timedelta
import pandas_datareader as pdr
# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


# #----------
# HUBER_LOSS_DELTA = 1.0

# HUBER_LOSS_W=1
#
# #----------
# def huber_loss(y_true, y_pred):
#     err = y_true - y_pred
#
#     cond = K.abs(err) < HUBER_LOSS_DELTA
#     L2 = 0.5 * K.square(err)
#     L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
#
#     loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(
#
#     return K.mean(loss)
    # return tf.losses.huber_loss(y_true,y_pred,weights=HUBER_LOSS_W,delta=HUBER_LOSS_DELTA, reduction=None)

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()
        # #of layer
        # here the input layer have statecnt state and 64 units and middle layer have 128 units , will be much faster if reduce layer but it will got little unstable
        #The Brain class encapsulates the neural network. Our problem is simple enough so we will use only one hidden layer of 64 neurons,
        # with ReLU activation function. The final layer will consist of only two neurons, one for each available action.
        # Their activation function will be linear. Remember that we are trying to approximate the Q function,
        #Instead of simple gradient descend, we will use a more sophisticated algorithm RMSprop, and Mean Squared Error (mse) loss function.


        model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=128, activation='relu', input_dim=64))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        # model.compile(loss='mse', optimizer=opt)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
    # def isFull(self):
    #     return len(self.samples) >= self.capacity

# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 400000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay

UPDATE_TARGET_FREQUENCY = 1000


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if (random.random() < self.epsilon) and not needrepaly:
            return random.randint(0, self.actionCnt - 1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[1][0] for o in batch])
        states_ = numpy.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), self.stateCnt))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3]

            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][numpy.argmax(p_[i])]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)
    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    exp = 0
    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass
# -------------------- Stock paramter ---------------------
class Stock:
    def __init__(self, investment,startdate,enddate,company):
        self.investment = investment
        self.date=startdate
        self.stepTimeDelta=timedelta(1)
        self.stockTimeDelta=timedelta(5)
        self.company=company
        self.transactionFee=TRANSACTION_FEE
        self.enddate=enddate
        self.isCached=False
        self.maxindex=float('+inf')
        self.cacheStock= None
        # self.env = gym.make(problem)
    def calcReward(self,cash,stock,price):
        return cash+stock*price

    def stockTransaction(self,cash,stock,index,action):
        # action : sell, buy , hold
        stockPrice=float(self.closePrice(self.company,index))
        currentValue=self.calcReward(cash,stock,stockPrice)
        if action == 2 :
            return [cash,stock,stockPrice],self.calcReward(cash,stock,stockPrice)
        if action == 1:
            return [0,stock+(cash-self.transactionFee)/stockPrice,stockPrice],self.calcReward(0,stock+(cash-self.transactionFee)/stockPrice,stockPrice)
        if action ==0:
            return [cash+stock*stockPrice-self.transactionFee,0,stockPrice],self.calcReward(cash+stock*stockPrice-self.transactionFee,0,stockPrice)

    # def step(self,s,a,date):
    #     if date+self.stepTimeDelta>self.enddate:
    #         return s,0,True,"pass today"
    #     # Cash, Stock ,current stock price
    #     s_,r=self.stockTransaction(s[0],s[1],date,a)
    #
    #     return s_,r,False,''
    def step(self,s,a,index):
        reward=self.calcReward(s[0],s[1],s[2])

        if index+1>self.maxindex:
            return s,reward,True,"pass today"
        if reward<STOP_INVERST_PREC*self.investment:
            print ('stop , only have %s percentage  investment'%(STOP_INVERST_PREC*100))
            return s,reward,True,"stop"
        # Cash, Stock ,current stock price
        s_,r=self.stockTransaction(s[0],s[1],index,a)

        return s_,r,False,''

    def closePrice(self,company,index):
        if not self.isCached:
            self.cacheStock=pdr.get_data_yahoo(symbols=company, start=self.date, end=self.enddate)['Close']
            self.maxindex=self.cacheStock.__len__()-1
            print ('start cache index for company %s from date %s to date %s ,total number of record %s')%(self.company,self.date.strftime('%Y-%m-%d'),self.enddate.strftime('%Y-%m-%d'),self.maxindex+1)
            self.isCached=True
        return self.cacheStock[index]

    def run(self, agent,printout=False):
        s=[self.investment,0,self.closePrice(self.company,0)]
        s=numpy.array(s)
        reward = self.investment
        index=0
        while True:
            # self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.step(s,a,index)
            s_ = numpy.array(s_)
            index+=1
            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            # use accumulated reward
            reward = r

            if done:

                break
            # print("step reward:", r)
        if printout:
            print("Total reward:", reward)
        MYLIST.append(reward)



# -------------------- MAIN ----------------------------
company = 'SAP'
# tartdate=datetime(2015,01,01)

# enddate=datetime.now()

startdate=datetime(2016,1,1)

enddate=datetime(2017,1,1)
# stop when you cash value is only 50% of your total cash

STOP_INVERST_PREC=float(0.5)
TRANSACTION_FEE=2
LEARNING_RATE = 0.00025
needrepaly=True
replayFile='stock_SAP_episode30955_2017-12-06.h5'
initalInvestment=1000
MYLIST=[]

#___________________________________ above is control parameter ________________________

stock = Stock(initalInvestment,startdate,enddate,company)

stateCnt = 3  # Cash, Stock ,current stock price
actionCnt = 3  # sell, buy , hold


agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)
count = 0
try:
    if needrepaly:
       print('use replay file %s'%(replayFile))
       agent.brain.model.load_weights(replayFile)

    else:
        while randomAgent.exp < MEMORY_CAPACITY:
            stock.run(randomAgent)
        print('use random agent to initial model %s tims'%(randomAgent.exp))

    agent.memory = randomAgent.memory
    randomAgent = None
    while True:
        count += 1
        print("episode # %s" % (count))
        stock.run(agent,printout=True)

finally:
    agent.brain.model.summary();

    agent.brain.model.save("stock_%s_episode%s_%s.h5"%(company,count,datetime.now().strftime('%Y-%m-%d')))
    with open("advv2_stock_%s_episode%s_%s.csv"%(company,count,datetime.now().strftime('%Y-%m-%d')), 'wb') as myfile:
        wr = csv.writer(myfile, delimiter="\n")
        wr.writerow(MYLIST)

