
import tensorflow as tf
import numpy as np

# 定义了 ?行 1列数据（方便传入一个参数 或 向量参数进行运算，自动根据输入参数判断维度）
x = tf.placeholder(tf.float32, [None, 1])

## 诡异了，谁来更新的W和b (GradientDescentOptimizer自动更新)
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_sum(tf.pow((y_-y), 2))


train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

init_op = tf.global_variables_initializer()

writer = tf.summary.FileWriter('./graphTwo', graph=tf.get_default_graph())
writer.close()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(100):
        xs = np.array([[i]])
        ys = np.array([[2*i]])

        feed = {x: xs, y_: ys}
        sess.run(train_step, feed_dict=feed)

        print('W: %f   , b: %f  ' % (sess.run(W), sess.run(b)))
