
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 1])

## 诡异了，谁来更新的W和b
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_sum(tf.pow((y_-y), 2))


train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    xs = np.random.randn(100, 1)
    ys = np.array(xs * 2)
    feed = {x: xs, y_: ys}

    for _ in range(100):    
        sess.run(train_step, feed_dict=feed)

        print('W: %f   , b: %f  ' % (sess.run(W), sess.run(b)))
