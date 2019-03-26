# !/usr/bin/python
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/22 10:01

import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('./flower_photos/sunflowers/23645265812_24352ff6bf.jpg', 'rb').read()
print(image_raw_data)

with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image_raw_data)
    print(image_data.eval())
    plt.imshow(image_data.eval())
    plt.show()
    print(image_data)
    # 图像的numpy格式转二进制保存到本地
    # encoded_image = tf.image.encode_jpeg(image_data)
    # with tf.gfile.GFile('./1.jpg', 'wb') as f:
    #     f.write(encoded_image.eval())

    img_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    # 调整图片大小，修改长宽尺寸
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    # plt.imshow(resized.eval())
    # plt.show()
    print(resized)
    # 以填充的方式改变大小，变形后图片小以全0填充图片到需要的大小
    # 图片大 自动截取原始图片中居中的部分
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)  # 图片的裁剪与填充
    plt.imshow(croped.eval())
    plt.show()
    # 按比例裁剪图片 比例是(0,1]
    centtal_cropped = tf.image.central_crop(img_data, 0.5)
    # tf.image.flip_left_right 左右翻转
    # tf.image.flip_up_down    上下翻转
    # tf.image.transpose_image 对角线翻转
    # tf.image.random_flip_left_right 50%的概率左右翻转
    # tf.image.random_flip_up_down
