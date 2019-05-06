import tensorflow as tf
import os
import numpy as np


def linear_regression():

    learning_rate = 0.01
    batch_size = 10

    X = tf.placeholder(tf.float32, [None, 1000])
    Y = tf.placeholder(tf.float32, [None, 4])

    W = tf.Variable(tf.zeros([1000, 4]))
    b = tf.Variable(tf.zeros([4]))

    logit_y = tf.add(tf.matmul(X, W), b)
    softmax_y = tf.nn.softmax(logit_y)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(softmax_y), reduction_indices=[1]))
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
    # print(train_data.shape)
    # print(train_data[:5])

    train_label = np.array(label_lines, np.int)
    train_label = train_label.reshape(-1, 4)
    # print(train_label.shape)
    # print(train_label[:5])

    s = np.arange(train_data.shape[0])
    np.random.shuffle(s)

    train_data = train_data[s]
    train_label = train_label[s]
    print(train_data.shape)
    # print(train_data[:5])
    print(train_label.shape)
    # print(train_label[:5])

    epoch = int(train_data.shape[0]/batch_size)

    for i in range(epoch):
        batch_xs = train_data[i:i+batch_size, :]
        batch_ys = train_label[i:i+batch_size, :]

        # print(batch_xs.shape)
        # print(batch_ys.shape)

        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

    f_data = open("test_set_data.txt", 'r', encoding='utf-8')
    f_label = open("test_set_label.txt", "r", encoding='utf-8')

    data_lines = f_data.read().split()
    f_data.close()

    label_lines = f_label.read().split()
    f_label.close()

    test_data = np.array(data_lines, np.float)
    test_data = test_data.reshape(-1, 1000)

    test_label = np.array(label_lines, np.int)
    test_label = test_label.reshape(-1, 4)

    correct_prediction = tf.equal(tf.argmax(softmax_y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("정확도: ", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))


if __name__ == "__main__":
    linear_regression()

