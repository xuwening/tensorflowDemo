import os
import tensorflow as tf

w = tf.Variable(tf.zeros([2,1]), name="weight")
b = tf.Variable(0., name="bias")

#计算腿短模型在数据x上的输出，并将结果返回
def inference(x):
    return tf.matmul(x, w) + b

# 依据训练数据x及其期望输出y，计算损失
def loss(x, y):
    y_predicted = inference(x)
    return tf.reduce_sum(tf.squared_difference(y, y_predicted))

#读取或生成训练数据x机器期望输出y
def inputs():
    weight_age = [
        [84,  46],
        [73,  20],
        [65,  52],
        [70,  30],
        [76,  57],
        [69,  25],
        [63,  28],
        [72,  36],
        [79,  57],
        [75,  44],
        [27,  24],
        [89,  31],
        [65,  52],
        [57,  23],
        [59,  60],
        [69,  48],
        [60,  34],
        [79,  51],
        [75,  50],
        [82,  34],
        [59,  46],
        [67,  23],
        [85,  37],
        [55,  40],
        [63,  30]
    ]

    blood_fat_content = [
        354,
        190,
        405,
        263,
        451,
        302,
        288,
        385,
        402,
        365,
        209,
        290,
        346,
        254,
        395,
        434,
        220,
        374,
        308,
        220,
        311,
        181,
        274,
        303,
        244
    ]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)

#依据计算的总损失训练或调整模型参数
def train(totao_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(totao_loss)

#对训练得到的模型进行评估
def evaluate(sess, x, y):
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))


saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    initial_step = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    x, y = inputs()

    total_loss = loss(x, y)
    train_op = train(total_loss)

    #多线程协调器
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #实际的训练迭代次数
    training_steps = 1000
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print("loss:", sess.run([total_loss]))
        if step %1000 == 0:
            saver.save(sess, './my-model', global_step=step)

    evaluate(sess, x, y)

    saver.save(sess, './my-model', global_step=training_steps)

    coord.request_stop()
    coord.join(threads)
    sess.close()

