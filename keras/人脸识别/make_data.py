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

'''
制作方向是数据类别少，单个数据数据量大
比如只有五六个人的照片，，每个人都有大量的数据集
关于类别多 单个数据量小的问题目前没有更多的时间考虑这种情况
有一个思路就是根据标签找类别组合成batch输入 这个需要修改训练生成器的方法

假设有五个人的数据 每个人的数据集很大，此方法比较笨重 要保证每个人的数据集量大小一样 ，并且都可以被40整除
这样保证不用标记类别 根据数据的位置就可以推算出属于哪个人 有利于训练验证生成器编写相对简单
可以被40整除 因为每次batch我会以20为基数从前向后提取待训练数据，
再根据位置从同类别的剩余数据中随机抽取20个数据与所有其他类别中随机抽取20个数据组组合成batch的输入
如果是采用随机梯度下降的话 就不需要单类别总数被40整除 因为还是通过位置确定类别 所以训练数据中每个类别总数要保持一致
'''


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
