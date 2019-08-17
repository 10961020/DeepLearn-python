#!/usr/bin/env python
# encoding: utf-8
# @author:tong.z
# @time: 2019/8/12 15:06

import os
import random
import numpy as np
from keras import layers
from keras import backend as k
from keras import initializers
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D, AveragePooling2D
from keras.utils import plot_model

'''
make_data里有数据集验证集的制作细节问题，因为三元损失计算方式特殊的缘故
https://github.com/michuanhaohao/keras_reid/blob/master/reid_tripletcls.py
https://blog.csdn.net/Lauyeed/article/details/79514839
'''


# TODO 上一层输出， 步长， 1*1深度， 3*3reduce深度， 3*3深度， 5*5reduce深度， 5*5深度， pool深度
def inception_v1(input_np, st, d1c1, d3c1, d3c3, d5c1, d5c5, b4c1):
    if d1c1:
        tower_1 = Conv2D(d1c1, (1, 1), activation='relu')(input_np)

    tower_2 = Conv2D(d3c1, (1, 1), activation='relu')(input_np)
    tower_2 = Conv2D(d3c3, (3, 3), strides=st, padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(d5c1, (1, 1), activation='relu')(input_np)
    tower_3 = Conv2D(d5c5, (5, 5), strides=st, padding='same', activation='relu')(tower_3)
    if b4c1:
        tower_4 = Conv2D(b4c1, (1, 1), activation='relu')(tower_3)
    else:
        tower_4 = MaxPooling2D((2, 2), strides=2)(input_np)

    if d1c1:
        return layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=-1)
    return layers.concatenate([tower_2, tower_3, tower_4], axis=-1)


# TODO 网络结构
def cownet():
    input_tensor = Input((229, 229, 3))
    x = input_tensor
    x = Conv2D(64, (7, 7), strides=2, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = inception_v1(x, 1, 64, 96, 128, 16, 32, 32)
    x = inception_v1(x, 1, 64, 96, 128, 32, 64, 64)
    x = inception_v1(x, 2, 0, 128, 256, 32, 64, 0)
    x = inception_v1(x, 1, 256, 96, 192, 32, 64, 128)
    x = inception_v1(x, 1, 224, 112, 224, 32, 64, 128)
    x = inception_v1(x, 1, 192, 128, 256, 32, 64, 128)
    x = inception_v1(x, 1, 160, 144, 288, 32, 64, 128)
    x = inception_v1(x, 2, 0, 160, 256, 64, 128, 0)
    x = inception_v1(x, 1, 384, 192, 384, 48, 128, 128)
    x = inception_v1(x, 1, 384, 192, 384, 48, 128, 128)
    x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.001),
              kernel_initializer=initializers.random_normal(stddev=0.01),
              activation='relu')(x)

    return Model(inputs=input_tensor, outputs=x)


# TODO 三元损失
def triplet_loss(y_true, y_pred):
    # y_pred = k.l2_normalize(y_pred, axis=1)
    # a = np.random.normal(loc=50, scale=20, size=256*60).reshape((60, 256))
    batch = 20
    ref1 = y_pred[0:batch]
    pos1 = y_pred[batch:batch+batch]
    neg1 = y_pred[batch+batch:3*batch]
    dis_pos = k.sqrt(k.sum(k.square(ref1 - pos1), axis=1, keepdims=True))
    dis_neg = k.sqrt(k.sum(k.square(ref1 - neg1), axis=1, keepdims=True))
    d1 = k.maximum(0.0, dis_pos - dis_neg + 2)
    return k.mean(d1)


# TODO 训练生成器
def gen_train(batch_size=20):
    processed_data = np.load('cow.npy')
    train_images = processed_data[0]
    class_num = processed_data[2]
    start_size = 0
    over_size = start_size+batch_size
    train_images_len = len(train_images)
    images_class_len = len(train_images) // class_num
    while True:
        if over_size > train_images_len:
            start_size = 0
            over_size = start_size + batch_size
        cow_img = train_images[start_size:over_size]
        cow_img.extend(random.sample(train_images[start_size//images_class_len*images_class_len:start_size] +
                                     train_images[over_size:(start_size//images_class_len+1)*images_class_len], batch_size))
        cow_img.extend(random.sample(train_images[:start_size//images_class_len*images_class_len] +
                                     train_images[(start_size//images_class_len+1)*images_class_len:], batch_size))

        train_np = np.array(cow_img, dtype=float)
        train_np /= 255
        y = np.zeros([batch_size*3, 1])
        start_size = over_size
        over_size = start_size + batch_size
        yield train_np, y


# TODO 验证生成器
def gen_val(batch_size=20):
    processed_data = np.load('cow.npy')
    val_images = processed_data[1]
    class_num = processed_data[2]
    start_size = 0
    over_size = start_size+batch_size
    val_len = len(val_images)
    images_class_len = len(val_images) // class_num
    while True:
        if over_size > val_len:
            start_size = 0
            over_size = start_size + batch_size
        cow_img = val_images[start_size:over_size]
        cow_img.extend(random.sample(val_images[start_size//images_class_len*images_class_len:start_size] +
                                     val_images[over_size:(start_size//images_class_len+1)*images_class_len], batch_size))
        cow_img.extend(random.sample(val_images[:start_size//images_class_len*images_class_len] +
                                     val_images[(start_size//images_class_len+1)*images_class_len:], batch_size))

        val_np = np.array(cow_img, dtype=float)
        val_np /= 255
        y = np.zeros([batch_size*3, 1])
        start_size = over_size
        over_size = start_size + batch_size
        yield val_np, y


if __name__ == '__main__':
    model = cownet()
    # print(model.summary())
    model.compile(loss=triplet_loss, optimizer=Adam(1e-3, decay=1e-6),
                  # metrics=['accuracy']
                  )
    best_model = ModelCheckpoint("cow_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
    model.fit_generator(gen_train(20), steps_per_epoch=20, epochs=40,
                        validation_data=gen_val(20), validation_steps=5,
                        callbacks=[best_model])


