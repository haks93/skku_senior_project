import tensorflow as tf
<<<<<<< HEAD
import os
import numpy as np
import matplotlib.pyplot as plt


def DNN():
    learning_rate = 0.01
    batch_size = 10

    X = tf.placeholder(tf.float32, [None, 1000])
    Y = tf.placeholder(tf.float32, [None, 4])

    W = tf.Variable(tf.zeros([1000, 4]))
    b = tf.Variable(tf.zeros([4]))

    logit_y = tf.add(tf.matmul(X, W), b)
    softmax_y = tf.nn.softmax(logit_y)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(softmax_y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    os.chdir("../data/txt/5.refined_rate")
    f_data = open("train_set_data.txt", 'r', encoding='utf-8')
    f_label = open("train_set_label.txt", "r", encoding='utf-8')

    data_lines = f_data.read().split()
    f_data.close()

    label_lines = f_label.read().split()
    f_label.close()

    train_data = np.array(data_lines, np.float)
    train_data = train_data.reshape(-1, 1000)

    train_label = np.array(label_lines, np.int)
    train_label = train_label.reshape(-1, 4)

    s = np.arange(train_data.shape[0])
    np.random.shuffle(s)

    train_data = train_data[s]
    train_label = train_label[s]
    print(train_data.shape)
    # print(train_data[:5])
    print(train_label.shape)
    # print(train_label[:5])

    n = train_data.shape[0]
    n2 = train_data.shape[1]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    model.add(tf.keras.layers.Dense(n2, input_shape=(n2,), activation='relu'))
    # model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
    # model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
    # model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
    # model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
    # model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
    model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

    hist = model.fit(train_data, train_label, validation_data=(train_data, train_label), epochs=1)

    plt.figure(figsize=(12, 8))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
    plt.show()


if __name__ == "__main__":
    DNN()
=======
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
>>>>>>> e040395e9e211d1ae06791d491b030eeff5c3c30
