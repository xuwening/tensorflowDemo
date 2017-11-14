
#图的应用

import tensorflow as tf

graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant(2)
    b = tf.constant(2)
    output = tf.add(a, b)

with tf.Session(graph=graph1) as sess:
    print(sess.run(output))

graph2 = tf.Graph()
with graph2.as_default():
    a = tf.constant(6)
    b = tf.constant(2)
    output = tf.div(a, b)

with tf.Session(graph=graph2) as sess:
    print(sess.run(output))
