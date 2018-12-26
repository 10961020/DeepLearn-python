#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     :2018/12/20 9:14
# @Author   :tong.z
'''
无锡所比赛所使用的对测试集的验证分类
'''
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import numpy as np

# base_dir = '/data/Deeplearn/wusun/1'  # 保存较小数据集的目录
# train_dir = os.path.join(base_dir, 'train')
# # os.mkdir(train_dir)
# validation_dir = os.path.join(base_dir, 'validation')
# # os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# # os.mkdir(test_dir)
#
# #test_datagen = ImageDataGenerator(rescale=1./255)
model = load_model('car_small_3.h5')
#
# #test_generator = test_datagen.flow_from_directory(test_dir,
# #                                                  target_size=(200, 200),
# #                                                  batch_size=20,
# #                                                  class_mode='categorical')
test_dir = ' '
for root, dirs, files in os.walk(test_dir):
    for jpg_path in files:
        img = image.load_img(root + '/' + jpg_path, target_size=(200, 200))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255
        # model.predict(img_tensor)
        i = model.predict(img_tensor)
        # print(np.where(i == np.max(i))[1])
        # print(i[1])
        j = (int(i[1])+2) % 4     # label 装换
        with open('result.txt', "a") as f:
            f.write(str(j) + '#' + jpg_path+'\n')
'''
半遮挡 全遮挡 未悬挂 正常
未悬挂 正常 部分遮挡 完全遮挡
'''
# test_loss,test_acc = model.evaluate_generator(test_generator, steps=50)
# print('test acc:', test_acc)
