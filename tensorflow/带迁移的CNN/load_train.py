# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/19 13:43

import glob
import os.path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

input_data = 'flower_processed_data.npy'
train_file = './save_model'
ckpt_file = 'inception_v3_2016_08_28/inception_v3.ckpt'

learning_rate = 0.0001
steps = 300
batch = 8
n_classes = 5

checkpoint_exclude_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'
trainable_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'


def get_tuned_variables():
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(argv=None):
    processed_data = np.load(input_data)
    training_images = processed_data[0]
    training_labels = processed_data[1]
    n_training_example = len(training_labels)
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print('%d training examples, %d validation examples and %d testing examples.'
          % (n_training_example, len(validation_labels), len(testing_labels)))

    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], 'labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=n_classes)

    training_variables = get_trainable_variables()
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, n_classes), logits, weights=1.0)
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.losses.get_total_loss())

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    load_fn = slim.assign_from_checkpoint_fn(ckpt_file, get_tuned_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('loading tuned variables from %s' % ckpt_file)
        load_fn(sess)

        start = 0
        end = batch
        for i in range(steps):
            sess.run(train_step, feed_dict={
                images: training_images[start:end],
                labels: training_labels[start:end]})

            if i % 30 == 0 or i+1 == steps:
                saver.save(sess, train_file, global_step=i)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels})
                print('step %d: Vaildation accuracy = %.1f%%' % (i, validation_accuracy*100.0))
            if start == n_training_example:
                start = 0
            end = start + batch
            if end > n_training_example:
                end = n_training_example

        test_accuary = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuary*100.0))

if __name__ == '__main__':
    tf.app.run()
