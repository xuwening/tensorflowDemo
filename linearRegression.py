
import os
import tensorflow as tf

W = tf.Variable(tf.zeros([2,1]), name="weights")
b = tf.Variable(0., name="bias")

def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, y_predicted))


def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))

# saver = tf.train.Saver()

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    init_step  = 0

    # ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     init_step = int(ckpt.model_checkpint_path.rsplit('-', 1)[1])

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    # writer = tf.train.SummaryWriter('./graphOne', graph=tf.get_default_graph())
    writer = tf.summary.FileWriter('./graphOne', graph=tf.get_default_graph())
    writer.close()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    training_steps = 1000
    for step in range(init_step, training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print('loss:', sess.run([total_loss]))

        # if step % 1000 == 0:
        #     saver.save(sess, 'my_model', global_step=step)
    
    evaluate(sess, X, Y)

    # saver.save(sess, 'my_model', global_step=training_steps)

    coord.request_stop()
    coord.join(threads)
