
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(2)
output = tf.add(a, b)

sess = tf.Session()
print(sess.run(output))