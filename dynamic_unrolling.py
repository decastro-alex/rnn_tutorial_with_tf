import tensorflow as tf
import numpy as np
from static_unrolling import N_INPUTS, N_NEURONS, N_STEPS

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_STEPS, N_INPUTS])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=N_NEURONS)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

X_batch = np.array([
    # t=0       t=1
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]]
])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output_val = outputs.eval(feed_dict={X: X_batch})

#print(output_val)

if __name__ == '__main__':
    print("not doing anything now")