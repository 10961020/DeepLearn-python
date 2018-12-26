# !/usr/bin/python  
# encoding: utf-8
# author: zhangtong

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import matplotlib.pyplot as plt
import os


base_dir = '/data/Deeplearn/wusun/1'  # 保存较小数据集的目录
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(200, 200, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
conv_base.trainable = False
model.summary()

model.compile(loss='categorical_crossentropy',               # 使用的损失函数 二元交叉熵
              optimizer=optimizers.RMSprop(lr=2e-5),    # 使用的优化器 RMS
              metrics=['acc'])                          # 监控精度

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 图像乘以1./255 缩放到0-1之间
    rotation_range=40,          # 角度值，图像随机旋转的角度范围
    width_shift_range=0.2,      # 水平方向上平移的范围
    height_shift_range=0,     # 垂直方向上平移的范围
    shear_range=0.2,            # 随机错切变换的角度
    zoom_range=0.2,             # 图像随机缩放的范围
    horizontal_flip=True)       # 随机将一半图像水平翻转
test_datagen = ImageDataGenerator(rescale=1./255)  # 图像增强不用于验证集

train_generator = train_datagen.flow_from_directory(
    train_dir,                  # 训练集路径
    target_size=(200, 200),     # 将所有图像的大小调整为150*150
    batch_size=64,              # 每次批量大小
    class_mode='categorical')        # 因为使用的二元交叉熵
validation_generator = train_datagen.flow_from_directory(
    validation_dir,             # 验证集路径
    target_size=(200, 200),
    batch_size=64,
    class_mode='categorical')

history = model.fit_generator(              # 开始训练
    train_generator,                        # 训练模型使用的训练集图像生成器
    steps_per_epoch=281,                    # 从生成器中抽取 steps_per_epoch 个批量后，进入下次迭代
    epochs=70,                              # 迭代次数
    validation_data=validation_generator,   # 验证集
    validation_steps=50)                    # 从验证集里抽取多少个批量用于评估

model.save('car_small_3.h5')      # 保存训练好的模型参数

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# epochs = range(1, len(acc)+1)
#
# plt.plot(epochs, acc, 'bo', label='Training_acc')
# plt.plot(epochs, val_acc, 'b', label='validation_acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# 
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training_loss')
# plt.plot(epochs, val_loss, 'b', label='validation_loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1' or layer.name == 'block4_conv1':
        set_trainable = True
    if set_trainable:
        layers.trainable = True
    else:
        layers.trainable = False
model.summary()
model.compile(loss='categorical_crossentropy',               # 使用的损失函数 二元交叉熵
              optimizer=optimizers.RMSprop(lr=1e-5),    # 使用的优化器 RMS
              metrics=['acc'])                          # 监控精度

history = model.fit_generator(              # 开始训练
    train_generator,                        # 训练模型使用的训练集图像生成器
    steps_per_epoch=281,                    # 从生成器中抽取 steps_per_epoch 个批量后，进入下次迭代
    epochs=70,                              # 迭代次数
    validation_data=validation_generator,   # 验证集
    validation_steps=50)                    # 从验证集里抽取多少个批量用于评估

model.save('car_small_4.h5')

history = model.fit_generator(              # 开始训练
    train_generator,                        # 训练模型使用的训练集图像生成器
    steps_per_epoch=55,                    # 从生成器中抽取 steps_per_epoch 个批量后，进入下次迭代
    epochs=100,                              # 迭代次数
    validation_data=validation_generator,   # 验证集
    validation_steps=50)                    # 从验证集里抽取多少个批量用于评估

model.save('car_small_56789.h5')
