
import tensorflow as tf
import os

W = tf.Variable(tf.zeros([4,3]), name="weights")
b = tf.Variable(tf.zeros([3]), name="bias")

def combine_inputs(x):
    return tf.matmul(x, W), b

def inference(x):
    return tf.nn.softmax(combine_inputs(x))

def loss(x, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(combine_inputs(x), y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + '/' + file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded,
                                batch_size=batch_size,
                                capacity=batch_size * 50,
                                min_after_dequeue=batch_size)


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label = \
    read_csv(100, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])

    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.pack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, [:Iris-virginica])
    ]))))

    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))
    return features, label_number


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, x, y):
    predicated = tf.cast(tf.arg_max(inference(x), 1), tf.int32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicated, y), tf.float32))))

with tf.Session() as sess:

    tf.global_variable_initializer().run()

    x, y = inputs()

    total_loss = loss(x, y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    trainning_steps = 1000
    for step in range(trainning_steps):
        sess.run([train_op])

        if step % 10 == 0:
            print('loss: ', sess.run([total_loss]))

    evaluate(sess, x, y)

    coord.request_stop()
    coord.join(threads)