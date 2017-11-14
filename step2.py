
# tensorflow执行demo

import tensorflow as tf

matrix1 = tf.constant([[11, 12], [21, 22]])
matrix2 = tf.constant([[31, 32], [41, 42]])

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
print(sess.run(product))
