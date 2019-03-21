# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/18 10:08


import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

input_data = './flower_photos'
output_file = ''

validation_percentage = 10
test_percentage = 10


# 读取数据集并分割为训练数据，验证数据和测试数据
def create_image_lists(sess, testing_percentage, validating_percentage):
    sub_dirs = [x[0]for x in os.walk(input_data)]
    print(sub_dirs)
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0
    # 遍历子路径
    for sub_dir in sub_dirs:
        if is_root_dir:  # 根路径直接跳过
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg',]
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(sub_dir, '*'+extension)
            file_list.extend(glob.glob(file_glob))  # glob库很强
        if not file_list: continue

        for file_name in file_list:
            print(file_name)
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            chance = np.random.randint(100)  # 真随机  8 1 1比例分配训练验证测试集
            if chance < validating_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (validating_percentage+test_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    state = np.random.get_state()  # 数据集随机打乱以获取更好的训练效果
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, test_percentage, validation_percentage)
        np.save(output_file, processed_data)


if __name__ == '__main__':
    main()
