# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/1/15 15:47
'''
    根据已经有的模型.h5文件 继续进行训练样例
'''
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import matplotlib.pyplot as plt
import os


base_dir = '/data/Deeplearn/wusun/2/'  # 小数据集的路径
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

model = load_model('wusun_1.h5')  # 打开本地模型 使用此模型继续训练
# model.trainable = True
set_trainable = False
for layer in model.layers:
    layer.summary()
    for layer_1 in layer.layers:
        print(layer_1)
        if layer_1.name == 'block5_conv1':
            break
        layer_1.trainable = False
    break
model.compile(loss='categorical_crossentropy',               # 使用的损失函数 二元交叉熵
              optimizer=optimizers.RMSprop(lr=1e-5),    # 使用的优化器 RMS
              metrics=['acc'])                          # 监控精度
model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 图像乘以1./255 缩放到0-1之间
    rotation_range=40,          # 角度值，图像随机旋转的角度范围
    width_shift_range=0.2,      # 水平方向上平移的范围
    height_shift_range=0,       # 垂直方向上平移的范围
    shear_range=0.2,            # 随机错切变换的角度
    zoom_range=0.2,             # 图像随机缩放的范围
    horizontal_flip=True)       # 随机将一半图像水平翻转
test_datagen = ImageDataGenerator(rescale=1./255)  # 图像增强不用于验证集

train_generator = train_datagen.flow_from_directory(
    train_dir,                  # 训练集路径
    target_size=(400, 400),     # 将所有图像的大小调整为150*150
    batch_size=64,              # 每次批量大小
    class_mode='categorical')        # 因为使用的二元交叉熵
validation_generator = test_datagen.flow_from_directory(
    validation_dir,             # 验证集路径
    target_size=(400, 400),
    batch_size=64,
    class_mode='categorical')

history = model.fit_generator(              # 开始训练
    train_generator,                        # 训练模型使用的训练集图像生成器
    steps_per_epoch=72,                     # 从生成器中抽取 steps_per_epoch 个批量后，进入下次迭代
    epochs=70,                              # 迭代次数
    validation_data=validation_generator,   # 验证集
    validation_steps=31)                    # 从验证集里抽取多少个批量用于评估

model.save('wusun_2.h5')


