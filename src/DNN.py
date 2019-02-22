import tensorflow as tf
from datetime import datetime
import pandas as pd
import numpy as np

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

data = pd.read_csv("../data/txt/3.nouns/korean.txt", sep=" ", header=None)
X_train = data.iloc[:5000]

n_input = X_train.shape[0]
n_output = 8
n_hidden1 = 5000
n_hidden2 = 5000
X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")


# def neuron_layer(X, n_neurons, name, activatioin=None):
#     with tf.name_scope(name):
#         n_input = int(X.get_shape()[1])
#         stddev = 2 / np.sqrt(n_input + n_neurons)
#         init = tf.truncated_normal((n_input, n_neurons), stddev=stddev)
#         W = tf.Variable(init, name="kernel")
#         b = tf.Variable(tf.zeros([n_neurons]), name="bias")
#         Z = tf.matmul(X, )
#         if activatioin is not None:
#             return activatioin(Z)
#         else:
#             return Z

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_output, name="output")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# with tf.Session() as sess:
#


# mse_summary = tf.summary.scalar('MSE', mse)
# file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
