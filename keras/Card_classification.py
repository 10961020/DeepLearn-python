# !/usr/bin/python  
# encoding: utf-8
# author: zhangtong
'''
此模型用于无锡科研所 道路车辆图像特征提取人工智能识别算法竞赛参赛 的比赛
比赛课题为 污损遮挡号牌检测

本人写的第一个神经网络模型 参考<python深度学习>这本书
因为第一次使用keras 原谅我本文中许多参数注释 
导入VGG16 去除模型的全连接层 使用迁移学习增加一个拥有64个神经元隐藏层和一个4维向量的输出层 (尝试过128，256神经元的隐藏层，验证集结果一直处于80%精度，训练已经达到98%)
思考很久应该是数据太少 以及还有错误数据的缘故

训练迭代140次(比赛时间太短，当时能使用的只有一个1080Ti的GPU)的测试成绩87.56 四类图片各十张
半遮挡 全遮挡 部分遮挡 三类测试数据查准率为100%，正常图片查准率为40% 五张分到全遮挡 一张分到部分遮挡
由于比赛方正常图片数据样本8W张 其他数据加起来不到一万，以及需要大量的数据清洗(数据中错误图片很多 包括分类错误、无关图片等)比较侧重于对其他三类数据的照顾
使用的训练集 比较少 正常2000左右 其他加起来2000左右 测试结果对正常图片拟合不够 下次注意 
跟同组讨论交流学习到清华教授的一些关于对数据清理的见解，茅塞顿开。奈何当时没有清理的条件，没能使用 很可惜

总结一下 数据清理的不够好 导致模型没能可以愉快学习特征 一块好的GPU真的很重要 GPU真的是越多越好能同时跑多个模型 效率提高


最新总结 验证集不能做数据增强！！！！！！！！！   嘴上说着不能  代码居然给增强了
'''
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
validation_generator = test_datagen.flow_from_directory(
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

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# 
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
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',               # 使用的损失函数 二元交叉熵
              optimizer=optimizers.RMSprop(lr=1e-5),    # 使用的优化器 RMS
              metrics=['acc'])                          # 监控精度
model.summary()
history = model.fit_generator(              # 开始训练
    train_generator,                        # 训练模型使用的训练集图像生成器
    steps_per_epoch=281,                    # 从生成器中抽取 steps_per_epoch 个批量后，进入下次迭代
    epochs=70,                              # 迭代次数
    validation_data=validation_generator,   # 验证集
    validation_steps=50)                    # 从验证集里抽取多少个批量用于评估

model.save('car_small_4.h5')
