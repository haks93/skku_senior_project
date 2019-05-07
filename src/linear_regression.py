import tensorflow as tf
import os
import numpy as np


def linear_regression(learning_rate, batch_size, rank):

    X = tf.placeholder(tf.float32, [None, rank])
    Y = tf.placeholder(tf.float32, [None, 4])

    W = tf.Variable(tf.zeros([rank, 4]))
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
    train_data = train_data.reshape(-1, rank)
    # print(train_data.shape)
    # print(train_data[:5])

    train_label = np.array(label_lines, np.int)
    train_label = train_label.reshape(-1, 4)
    # print(train_label.shape)
    # print(train_label[:5])

    idx_1 = idx_2 = idx_3 = 0
    for row in range(train_label.shape[0]):
        if (train_label[row] == [0, 1, 0, 0]).all() and idx_1 == 0:
            idx_1 = row
        elif (train_label[row] == [0, 0, 1, 0]).all() and idx_2 == 0:
            idx_2 = row
        elif (train_label[row] == [0, 0, 0, 1]).all() and idx_3 == 0:
            idx_3 = row

    print("## The Number of Train Set ##")
    '''
    print("Total:      ", train_label.shape[0])
    print("English:    ", idx_1)
    print("Korean:     ", idx_2 - idx_1)
    print("Mathematics:", idx_3 - idx_2)
    print("Science:    ", train_label.shape[0] - idx_3, "\n")
    '''

    print(train_label.shape[0])
    print(idx_1)
    print(idx_2 - idx_1)
    print(idx_3 - idx_2)
    print(train_label.shape[0] - idx_3, "\n")

    s = np.arange(train_data.shape[0])
    np.random.shuffle(s)

    train_data = train_data[s]
    train_label = train_label[s]
    '''
    print(train_data.shape)
    print(train_data[:5])
    print(train_label.shape)
    print(train_label[:5])
    '''

    epoch = int(train_data.shape[0]/batch_size)

    for i in range(epoch):
        batch_xs = train_data[i:i+batch_size, :]
        batch_ys = train_label[i:i+batch_size, :]

        '''
        print(batch_xs.shape)
        print(batch_ys.shape)
        '''

        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

    f_data = open("test_set_data.txt", 'r', encoding='utf-8')
    f_label = open("test_set_label.txt", "r", encoding='utf-8')

    data_lines = f_data.read().split()
    f_data.close()

    label_lines = f_label.read().split()
    f_label.close()

    test_data = np.array(data_lines, np.float)
    test_data = test_data.reshape(-1, rank)

    test_label = np.array(label_lines, np.int)
    test_label = test_label.reshape(-1, 4)

    # test_set 과목별 분류하여 각 과목 정확도를 확인.
    idx_1 = idx_2 = idx_3 = 0
    for row in range(test_label.shape[0]):
        if (test_label[row] == [0, 1, 0, 0]).all() and idx_1 == 0:
            idx_1 = row
        elif (test_label[row] == [0, 0, 1, 0]).all() and idx_2 == 0:
            idx_2 = row
        elif (test_label[row] == [0, 0, 0, 1]).all() and idx_3 == 0:
            idx_3 = row

    print("## The Number of Test Set ##")
    '''
    print("Total:      ", test_label.shape[0])
    print("English:    ", idx_1)
    print("Korean:     ", idx_2 - idx_1)
    print("Mathematics:", idx_3 - idx_2)
    print("Science:    ", test_label.shape[0] - idx_3, "\n")
    '''
    print(test_label.shape[0])
    print(idx_1)
    print(idx_2 - idx_1)
    print(idx_3 - idx_2)
    print(test_label.shape[0] - idx_3, "\n")

    correct_prediction = tf.equal(tf.argmax(softmax_y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("## Accuracy ##")
    '''
    print("Total:      ", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    print("English:    ", sess.run(accuracy, feed_dict={X: test_data[:idx_1, :], Y: test_label[:idx_1, :]}))
    print("Korean:     ", sess.run(accuracy, feed_dict={X: test_data[idx_1:idx_2, :], Y: test_label[idx_1:idx_2, :]}))
    print("Mathematics:", sess.run(accuracy, feed_dict={X: test_data[idx_2:idx_3, :], Y: test_label[idx_2:idx_3, :]}))
    print("Science:    ", sess.run(accuracy, feed_dict={X: test_data[idx_3:, :], Y: test_label[idx_3:, :]}))
    '''
    print(sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    print(sess.run(accuracy, feed_dict={X: test_data[:idx_1, :], Y: test_label[:idx_1, :]}))
    print(sess.run(accuracy, feed_dict={X: test_data[idx_1:idx_2, :], Y: test_label[idx_1:idx_2, :]}))
    print(sess.run(accuracy, feed_dict={X: test_data[idx_2:idx_3, :], Y: test_label[idx_2:idx_3, :]}))
    print(sess.run(accuracy, feed_dict={X: test_data[idx_3:, :], Y: test_label[idx_3:, :]}))

    os.chdir("../../../src")


if __name__ == "__main__":

    learning_rate = 0.5
    batch_size = 10
    rank = 1000

    print("## Learning Rate:", learning_rate)
    print("## Batch Size:   ", batch_size)
    print("## Rank:   ", rank, "\n")
    for i in range(10):
        print("----------", i, "th test ----------")
        linear_regression(learning_rate, batch_size, rank)

