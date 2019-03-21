# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/21 11:18

import tensorflow as tf

# 创建一个reader来读取tfrecord文件中的样例
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['output_record.tfrecord'])

# 创建一个队列来维护文件列表  读取多个样例是 read_up_to
# _, serialized_example = reader.read(filename_queue)
_, serialized_example = reader.read_up_to(filename_queue, 10)

# 从文件中读入一个样例  解析多个样例是 parse_example
# features = tf.parse_single_example(
features = tf.parse_example(
    serialized_example,
    features={
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)})

image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

# 29-36行的数据是每次读取多个样例的时候需要的 单张图片的话把这些行注释，并且需要替换44行run里的参数改为上三行变量
batch_size = 2
capacity = 1000 + 3 * batch_size
image.set_shape([10, 784])
label.set_shape(10)
pixels.set_shape(10)
image_batch, label_batch, pixel_batch = tf.train.batch(
    [image, label, pixels], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取tfrecord文件中的一个样例，当所有样例都读完了之后，此样例中程序会再重头读取
    for i in range(10):
        image1, label1, pixel1 = sess.run([image_batch, label_batch, pixel_batch])
        print(image1.shape, label1, pixel1)
