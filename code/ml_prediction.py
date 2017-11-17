from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import process_data as pd
rng = numpy.random

# Parameters
learning_rate = 0.00000001
training_epochs = 1500
display_step = 50
model_size = 10

# Training Data
x_train, y_train = pd.get_train_data()
train_X = numpy.asarray(x_train)
train_Y = numpy.asarray(y_train)
x_eval, y_eval = pd.get_eval_data()
eval_X = numpy.asarray(x_eval)
eval_Y = numpy.asarray(y_eval)

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(tf.float64, [None, model_size])
Y = tf.placeholder(tf.float64, [None, model_size])



# ## linear model
# # Set model weights
# W = tf.get_variable("W", [model_size, model_size], dtype=tf.float64)
# b = tf.get_variable("b", [model_size], dtype=tf.float64)
# # Construct a linear model
# pred = tf.add(tf.matmul(X, W), b)
# # Mean squared error
# cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples) + 0*tf.nn.l2_loss(W) + 0*tf.nn.l2_loss(b)
# # Gradient descent
# #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

## DL model
# Model architecture parameters
model_size = 10
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 10
xavier_init = tf.contrib.layers.xavier_initializer()
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.get_variable("W1", [model_size, n_neurons_1], initializer=xavier_init, dtype=tf.float64)
bias_hidden_1 = tf.get_variable("b1", [n_neurons_1], initializer=xavier_init, dtype=tf.float64)
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.get_variable("W2", [n_neurons_1, n_neurons_2], initializer=xavier_init, dtype=tf.float64)
bias_hidden_2 = tf.get_variable("b2", [n_neurons_2], initializer=xavier_init, dtype=tf.float64)
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.get_variable("W3", [n_neurons_2, n_neurons_3], initializer=xavier_init, dtype=tf.float64)
bias_hidden_3 = tf.get_variable("b3", [n_neurons_3], initializer=xavier_init, dtype=tf.float64)
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.get_variable("W4", [n_neurons_3, n_neurons_4], initializer=xavier_init, dtype=tf.float64)
bias_hidden_4 = tf.get_variable("b4", [n_neurons_4], initializer=xavier_init, dtype=tf.float64)

# Output layer: Variables for output weights and biases
W_out = tf.get_variable("W_out", [n_neurons_4, n_target], initializer=xavier_init, dtype=tf.float64)
bias_out = tf.get_variable("b_out", [n_target], initializer=xavier_init, dtype=tf.float64)

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
pred = tf.add(tf.matmul(hidden_4, W_out), bias_out)

# Cost function
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/tf.cast(tf.shape(X)[0], tf.float64)

# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Initializers
sigma = 1
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):
        #     sess.run(optimizer, feed_dict={X: x, Y: y})
        sess.run(optimizer, feed_dict = {X: train_X, Y : train_Y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
            #       "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, '\n')
    eval_cost = sess.run(cost, feed_dict={X: eval_X, Y: eval_Y})
    print("eval cost=", eval_cost, '\n')
    # print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    time = range(189)
    # print ("index", past, (numpy.append(train_X[0], train_Y[0])))
    plt.plot(time, pd.get_whole_eval_data(), 'r-', label='True price')
    for i in range(169):
        # plt.plot(past, numpy.append(eval_X[i], eval_Y[i]), 'r-', label='True price')
        predicted = sess.run(pred, feed_dict={X: eval_X, Y: eval_Y})
        future = range(i, i+10)
        plt.plot(future, predicted[i], '-b', label='Predicted price')
    # plt.legend()
    plt.show()
