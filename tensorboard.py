
import tensorflow as tf

with tf.name_scope("scopeA"):
    a = tf.add(1, 2, name="addA")
    b = tf.multiply(a, 3, name="mulA")

with tf.name_scope("scopeB"):
    c = tf.add(4, 5, name="addB")
    d = tf.multiply(c, 6, name="mulB")

e = tf.add(b, d, name="output")

write = tf.summary.FileWriter('./tbdt')
graph = tf.get_default_graph()
write.close()

