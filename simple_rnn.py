import tensorflow as tf
import numpy as np

N_INPUTS = 3
N_NEURONS = 5

X0 = tf.placeholder(tf.float32, [None, N_INPUTS])
X1 = tf.placeholder(tf.float32, [None, N_INPUTS])

Wx = tf.Variable(tf.random_normal(shape=[N_INPUTS, N_NEURONS], seed=-1), dtype=tf.float32)
Wy = tf.Variable(tf.random_normal(shape=[N_NEURONS, N_NEURONS], seed=-1), dtype=tf.float32)
b  = tf.Variable(tf.zeros([1, N_NEURONS], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()


# Mini-batch:        instance 0, instance 1, instance 2, instance 3
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])  # t=0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])  # t=1


init = tf.global_variables_initializer() # how does it initialize the global variables?

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X0_batch})

#print(Y0_val) # output at t = 0

#print(Y1_val) # output at t = 1

if __name__ == '__main__':
    print("constants have been loaded")

