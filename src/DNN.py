import tensorflow as tf
from datetime import datetime
import test

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

word_list = test()
m = 1000

X = tf.placeholder(tf.int32, shape=(None, m), name="X")
y = tf.placeholder(tf.int32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

init = tf.global_variables_initializer()

f = tf.random_uniform(-1.0, 1.0)

with tf.Session() as sess:
    init.run()
    result = f.eval()

