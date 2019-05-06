import tensorflow as tf
import os
import numpy as np

learning_rate = 0.1
batch_size = 100

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

os.chdir("../data/txt/5.refined_rate/train_set")
for item in os.listdir(os.getcwd()):
    f = open(item, 'r', encoding='utf-8')
    lines = f.readlines()
    rates_str = lines[0].split(" ")[:-1]

    rates_float = np.array(rates_str, np.float)
    rates_float = rates_float.reshape(1, 1000)

    print(rates_float)

    subject = lines[1]
    subject_idx = np.array([0, 0, 0, 0], np.int)
    if subject == "english":
        subject_idx[0] = 1
    elif subject == "korean":
        subject_idx[1] = 1
    elif subject == "mathematics":
        subject_idx[2] = 1
    elif subject == "science":
        subject_idx[3] = 1
    else:
        print("error 1")
    subject_idx = subject_idx.reshape(1, 4)

    batch_xs = rates_float
    batch_ys = subject_idx

    print(batch_xs.shape)

    sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

correct_prediction = tf.equal(tf.argmax(softmax_y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print("정확도: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))