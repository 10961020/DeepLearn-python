# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/21 10:43
'''
    将数据存储为TFrecord格式
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 生成整数型的属性
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets('D:/1/project/21_project_for_tensorflow/MNIST_data',
                                  dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels  # 训练数据所对应的正确答案，可以作为一个属性保存再tfrecord中
pixels = images.shape[1]  # 训练数据的图片分辨率，这可以作为一个属性保存到tfrecord中
num_examples = mnist.train.num_examples
print(num_examples)

filename = 'output_record.tfrecord'  # 保存地址跟文件名
with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
        image_raw = images[index].tostring()  # 将图像矩阵转化成一个字符串
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': int64_feature(pixels),
            'label': int64_feature(np.argmax(labels[index])),
            'image_raw': bytes_feature(image_raw)}))  # 将一个样例转化为Example Protocol Buffter,并将所有信息写入这个结构中
        writer.write(example.SerializeToString())
