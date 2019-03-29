# !/usr/bin/python
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/22 10:01

import matplotlib.pyplot as plt
import tensorflow as tf
'''
    图片处理试样
'''
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)  # 调整亮度 [-max_delta,max_delta)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 调整饱和度 [lower,upper]
        image = tf.image.random_hue(image, max_delta=0.2)  # 调整色相 [-max_delta,max_delta] max_delta取值[0,0.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 调整对比度 [lower,upper]
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 调整饱和度 [lower,upper]
        image = tf.image.random_brightness(image, max_delta=32./255.)  # 调整亮度 [-max_delta,max_delta)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 调整对比度 [lower,upper]
        image = tf.image.random_hue(image, max_delta=0.2)  # 调整色相 [-max_delta,max_delta] max_delta取值[0,0.5)
    # 色彩调整的API可能导致像素的实数值超出0.0-1.0的范围，将其截断在0.0-1.0之间，否则图片可能无法正常可视化，作为训练集也可能受影响
    # 这一截断过程应当在所有处理完成之后进行
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, weight, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        print(bbox)
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),  # 寻找随机裁剪边框
                                                                      bounding_boxes=bbox,
                                                                      min_object_covered=0.6)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)  # 裁剪
    distorted_image = tf.image.resize_images(distorted_image, [height, weight], method=np.random.randint(4))  # 改变图片大小
    distorted_image = tf.image.random_flip_left_right(distorted_image)  # 随机左右翻转
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


image_raw_data = tf.gfile.FastGFile('./flower_photos/sunflowers/23645265812_24352ff6bf.jpg', 'rb').read()

with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    for i in range(6):
        reslut = preprocess_for_train(image_data, 299, 299, boxes)
        plt.imshow(reslut.eval())
        plt.figure()
    plt.show()
    # print(image_data.eval())
    # plt.imshow(image_data.eval())
    # plt.show()
    # print(image_data)

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
