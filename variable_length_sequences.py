from dynamic_unrolling import *

tf.reset_default_graph()

seq_length = tf.placeholder(tf.int32, [None])

X = tf.placeholder(tf.float32, [None, N_STEPS, N_INPUTS]) # having a hard time visualizing this?

outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)

X_batch = np.array([
    # t=0       t=1
    [[0, 1, 2], [9, 8, 7]],  # instance 0
    [[3, 4, 5], [0, 0, 0]],  # instance 1 (padded with a zero vector)
    [[6, 7, 8], [6, 5, 4]],  # instance 2
    [[9, 0, 1], [3, 2, 1]]   # instance 3
])

seq_length_batch = np.array([2, 1, 2, 2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

print('decoded values:\n', output_val)
print('cell states:\n', states_val)

if __name__ == '__main__': # boiler plate code
    print("constants have been loaded")