from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import process_data as pd
import actions

class LSTM_model:
    def __init__(self):
        # Parameters
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = 0.0001
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 50, 0.1, staircase=True)
        self.training_epochs = 100
        self.display_step = 50
        self.model_size = 10
        self.lstm_size = 500
        self.n_target = 1

    def predict(self, company, start_year, start_month, start_day, end_year, end_month, end_day):
        # Training Data
        self.x_train, self.y_train, self.real_train = pd.get_train_data()
        self.train_X = numpy.expand_dims(numpy.asarray(self.x_train), axis=2)
        self.train_Y = numpy.asarray(self.y_train)
        self.x_eval, self.y_eval, self.real_eval = pd.get_eval_data()
        self.eval_X = numpy.expand_dims(numpy.asarray(self.x_eval), axis=2)
        self.eval_Y = numpy.asarray(self.y_eval)
        self.n_samples = self.train_X.shape[0]

        # tf Graph Input
        self.X = tf.placeholder(tf.float64, [None, self.model_size, 1],name='X')
        self.Y = tf.placeholder(tf.float64, [None, ],name='Y')
        self.lr = tf.placeholder(tf.float64, name='lr')

        ## LSTM model
        cell = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size, forget_bias=1.0, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs = self.X, dtype=tf.float64)
        state = tf.concat([states[0], states[1]], 1)

        xavier_init = tf.contrib.layers.xavier_initializer()
        # Output layer: Variables for output weights and biases
        W_out = tf.get_variable("W_out", [2*self.lstm_size, self.n_target], initializer=xavier_init, dtype=tf.float64)
        bias_out = tf.get_variable("b_out", [self.n_target], initializer=xavier_init, dtype=tf.float64)
        self.pred = tf.add(tf.matmul(state, W_out), bias_out)

        # Cost function
        # cost = tf.reduce_sum(tf.pow(pred-Y, 2))##/tf.cast(tf.shape(X)[0], tf.float64)
        cost = tf.nn.l2_loss(self.pred - self.Y)/tf.cast(tf.shape(self.X)[0], tf.float64)
        # Optimizer
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(cost)
        grads_and_vars = opt.compute_gradients(cost)

        # Initializers
        init = tf.global_variables_initializer()


        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Fit all training data
            for epoch in range(self.training_epochs):
                learning_rate = self.starter_learning_rate * (0.5 ** (epoch/100))
                sess.run(optimizer, feed_dict = {self.X: self.train_X, self.Y : self.train_Y, self.lr: learning_rate})

                # Display logs per epoch step
                if (epoch+1) % self.display_step == 0:
                    c = sess.run(cost, feed_dict={self.X: self.train_X, self.Y:self.train_Y, self.lr: learning_rate})
                    eval_cost = sess.run(cost, feed_dict={self.X: self.eval_X, self.Y: self.eval_Y, self.lr: learning_rate})
                    true_data = self.train_Y
                    predict_data = sess.run(self.pred, feed_dict={self.X: self.train_X, self.Y:self.train_Y, self.lr: learning_rate})
                    precision, _ = pd.judgeHighLow(true_data, predict_data)
                    gradients = sess.run(grads_and_vars, feed_dict={self.X: self.train_X, self.Y:self.train_Y, self.lr: learning_rate})
                    print("------ Epoch:", '%04d' % (epoch + 1), "training cost=", "{:.9f}".format(c), "eval cost=",
                          "{:.9f}".format(eval_cost), "the right call percentage is", precision, "learning rate is", sess.run(self.lr, feed_dict={self.X: self.train_X, self.Y:self.train_Y, self.lr: learning_rate}), "------")
            print("Optimization Finished!")
            # Graphic display
            true_data = self.real_train
            predict_data = sess.run(self.pred, feed_dict={self.X: self.train_X, self.Y: self.train_Y, self.lr: learning_rate})
            pd.plotPrices(true_data, predict_data, self.train_Y)

            x_eval, y_eval, real_eval = pd.get_data(company, start_year, start_month, start_day, end_year, end_month, end_day, model_size=self.model_size)
            x_eval = numpy.expand_dims(numpy.asarray(x_eval), axis=2)
            y_eval = numpy.asarray(y_eval)
            true_data = real_eval
            time = range(len(true_data))
            predict_data = sess.run(self.pred, feed_dict={self.X: x_eval, self.Y: y_eval, self.lr: learning_rate})
            pd.plotPrices(true_data, predict_data, y_eval)
            predict_true = [predict_data[i] * true_data[i] / y_eval[i] for i in time]
            return predict_true

model = LSTM_model()
## This following example showed how to call the predict function to predict the stock price between two dates 1/1/2017 and 10/1/2017
print ("the output is", model.predict('SAP', 2017, 1, 1, 2017, 10, 1))