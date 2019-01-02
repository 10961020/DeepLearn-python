#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     :2018/12/20 9:14
# @Author   :tong.z

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import shutil
import time
import numpy as np

base_dir = '/data/Deeplearn/wusun/1'  # 保存较小数据集的目录
train_dir = os.path.join(base_dir, 'train')
# # os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# # os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# # os.mkdir(test_dir)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
model = load_model('fog_9770.h5')
#
# test_generator = test_datagen.flow_from_directory(test_dir,
#                                                   target_size=(200, 200),
#                                                   batch_size=20,
#                                                   class_mode='binary')

# test_loss,test_acc = model.evaluate_generator(test_generator, steps=50)
# print('test acc:', test_acc)


test_dir_path = 'D:/1/project/photo_fog'
train_dir_1 = os.path.join(test_dir_path, 'fog')
train_dir_2 = os.path.join(test_dir_path, '2')
train_dir_3 = os.path.join(test_dir_path, 'no_fog')
if not os.path.exists(train_dir_1):
    os.makedirs(train_dir_1)
if not os.path.exists(train_dir_2):
    os.makedirs(train_dir_2)
if not os.path.exists(train_dir_3):
    os.makedirs(train_dir_3)
for root, dirs, files in os.walk(test_dir_path):
    for jpg_path in files:
        img = image.load_img(root + '/' + jpg_path, target_size=(200, 200))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255
        a = time.time()
        i = model.predict(img_tensor)
        # print(jpg_path + ': ' + str(1 if i[0][0] < 0.5 else 0))
        print(time.time()-a)
        if i[0][0] < 0.1:
            shutil.copy(os.path.join(test_dir_path, jpg_path), os.path.join(train_dir_1, jpg_path))
        elif i[0][0] < 0.9:
            print(jpg_path + ' 2: ' + str(i[0][0]))
            shutil.copy(os.path.join(test_dir_path, jpg_path), os.path.join(train_dir_2, jpg_path))
        else:
            print(jpg_path + ' 3: ' + str(i[0][0]))
            shutil.copy(os.path.join(test_dir_path, jpg_path), os.path.join(train_dir_3, jpg_path))
    break

