#!/usr/bin/env python
# encoding: utf-8
# @author:tong.z
# @time: 2019/8/11 15:10

import glob
import cv2
import os
import shutil
from keras import backend as k
import random
import shutil
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np


def create_image_lists():
    train_images = []
    val_images = []
    class_num = 0
    for path, dirs, files in os.walk('D:/project/face/human face'):
        len_images = len(files)
        if not len_images:
            continue
        print('file: {} len: {}'.format(os.path.basename(path), len_images))
        len_images = 20 * (int(len_images * .1) // 20)  # 验证测试集总数
        value = 0
        for file in files:
            # print(os.path.splitext(file)[0])
            img = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (229, 229))
            # img = np.array(img, dtype='float')
            # img /= 255

            if value == len_images:  # 保证train集总数可以被20整除
                train_images.append(img)
                continue

            chance = np.random.randint(100)  # 随机  8 1 1比例分配训练验证测试集
            if chance < 15:
                val_images.append(img)
                value += 1
            else:
                train_images.append(img)
        class_num += 1
    # np.random.shuffle(train_images)
    print('train_images length:\t', len(train_images))
    print('val_images length:\t', len(val_images))
    return np.asarray([train_images, val_images, class_num])


def main():
    processed_data = create_image_lists()
    np.save('cow.npy', processed_data)


main()
