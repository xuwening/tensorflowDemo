
# 逆概率
# y=Ax+B（A、B是常量），这是一条非常简单的数学方程式，有小学基础的人应该都知道。
# 我现在有很多的x和y值，所以问题就是如何通过这些x和y值来得到A和B的值？


import tensorflow as tf
import numpy as np


# 构造数据

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 建立tensorflow神经计算结构
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weight * x_data + biases


#判断与正确值的差距
loss = tf.reduce_mean(tf.square(y-y_data))

#根据差距进行反向传播修正参数
optimizer = tf.train.GradientDescentOptimizer(0.5)

#建立训练器
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#训练
for step in range(400):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weight), sess.run(biases))

