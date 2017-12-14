
import tensorflow as tf
import os

W = tf.Variable(tf.zeros([5,1]), name="weights")
b = tf.Variable(0., name="bias")

def combine_input(x):
    return tf.matmul(x, W) + b

def inference(x):
    return tf.sigmoid(combine_input(x))


def loss(x, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_input(x), labels=y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value =reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
    read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))

    return features, tf.reshape(survived, [100, 1])


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, x, y):
    predicted = tf.cast(inference(x) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))))

with tf.Session() as sess:

    tf.initialize_all_variables().run()

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

    import time
    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    