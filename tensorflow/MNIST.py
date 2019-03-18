# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/2/28 9:22

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 128

LEARNING_RATE_BASE = 0.8  # 学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减度
REGULARIZATION_RATE = 0.0001  # 正则化系数
TRAINING_STEPS = 30000  # 训练次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减度


def inference(input_tensor, avg_class, weights_1, biases_1, weights_2, biases_2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights_1)+biases_1)
        return tf.matmul(layer1, weights_2) + biases_2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights_1)+avg_class.average(biases_1)))
        return tf.matmul(layer1, avg_class.average(weights_2))+avg_class.average(biases_2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weight1, biases1, weight2, biases2)
    global_step = tf.Variable(0, trainable=False)  # trainable设置为不训练的变量

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight1) + regularizer(weight2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, variable_averages_op)
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), val acc is %g' % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test acc is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('../21_project_for_tensorflow/MNIST_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
