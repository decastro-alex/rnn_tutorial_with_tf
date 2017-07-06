import tensorflow as tf
import numpy as np
from simple_rnn import N_NEURONS, N_INPUTS

X0 = tf.placeholder(tf.float32, [None, N_INPUTS])
X1 = tf.placeholder(tf.float32, [None, N_INPUTS])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=N_NEURONS)
output_seqs, states = tf.contrib.rnn.static_rnn(
    basic_cell, [X0, X1], dtype=tf.float32) # how to initialize the weights?
Y0, Y1 = output_seqs

# Mini-batch:        instance 0, instance 1, instance 2, instance 3
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t=0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

init = tf.global_variables_initializer() # how does it initialize the global variables?

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X0_batch})

print(Y0_val) # output at t = 0

print(Y1_val) # output at t = 0