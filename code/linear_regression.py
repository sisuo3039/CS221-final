from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import process_data as pd
rng = numpy.random

# Parameters
learning_rate = 0.0000001
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

# Set model weights
W = tf.get_variable("W", [model_size, model_size], dtype=tf.float64)
b = tf.get_variable("b", [model_size], dtype=tf.float64)

# Construct a linear model
pred = tf.add(tf.matmul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/tf.cast(tf.shape(X)[0], tf.float64) #+ 0*tf.nn.l2_loss(W) + 0*tf.nn.l2_loss(b)
# cost = tf.nn.l2_loss(pred - Y)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
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

    # print ("the evaluation data at 0 is ", eval_X[0], eval_Y[0])
    # Graphic display
    # past = range(1, 21)

    time = range(189)
    # print ("index", past, (numpy.append(train_X[0], train_Y[0])))
    plt.plot(time, pd.get_whole_eval_data(), 'r-', label='True price')
    for i in range(169):
        # plt.plot(past, numpy.append(eval_X[i], eval_Y[i]), 'r-', label='True price')
        future = range(i, i+10)
        plt.plot(future, eval_X[i]. dot(sess.run(W)) + sess.run(b), '-b', label='Predicted price')
    # plt.legend()
    plt.show()


    true_data = eval_Y
    predict_data = eval_X. dot(sess.run(W))
