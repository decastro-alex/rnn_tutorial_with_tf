import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

## data generator
t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution) # initializing t0 randomly
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1) # what's the difference here?

t = np.linspace(t_min, t_max, (t_max - t_min) // resolution)
train_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.title("A time series (generated)", fontsize=14)
# plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
# plt.plot(train_instance[:-1], time_series(train_instance[:-1]), "b-", linewidth=3, label="A training instance")
# plt.legend(loc="lower left", fontsize=14)
# plt.axis([0, 30, -17, 13])
# plt.xlabel("Time")
# plt.ylabel("Value")
#
# plt.subplot(122)
# plt.title("A training instance", fontsize=14)
# plt.plot(train_instance[:-1], time_series(train_instance[:-1]), "bo", markersize=10, label="instance")
# plt.plot(train_instance[1:], time_series(train_instance[1:]), "w*", markersize=10, label="target")
# plt.legend(loc="upper left")
# plt.xlabel("Time")
#
# plt.savefig("time_series_plot")
# plt.show()

## deployment phases

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
#     output_size=n_outputs)
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

## execution phase

n_iterations = 10000
batch_size = 50

with tf.Session() as sess:
    init.run()
    mse = []
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)  # fetch the next training batch
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = np.append(mse, loss.eval(feed_dict={X: X_batch, y: y_batch}))  # PEP80 doesn't like spaces
            print(iteration, "\tMSE:", mse[-1::])

    X_new = time_series(np.array(train_instance[:-1].reshape(-1, n_steps, n_inputs))) # why does tensor flow like this template of tensor?
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    print(y_pred)

plt.title("log-MSE sampled every 100 iters") # this reflects more the nature of the stochastic gradient
plt.plot(range(n_iterations // 100), np.log(mse))
plt.savefig("mse_vs_iteration_step.png")

plt.title("Testing the model", fontsize=14)
plt.plot(train_instance[:-1], time_series(train_instance[:-1]), 'bo', markersize=10, label="instance")
plt.plot(train_instance[1:], time_series(train_instance[1:]), "w*", markersize=10, label="target")
plt.plot(train_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.savefig("time_series_prediction_plot.png")


# # X_batch, y_batch = next_batch(1, n_steps) # why do I need such deep array?
# X_batch # deeper tensor?
# np.c_[X_batch[0]] # column vector

