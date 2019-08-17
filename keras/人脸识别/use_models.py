#!/usr/bin/env python
# encoding: utf-8
# @author:tong.z
# @time: 2019/8/12 10:21

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


# TODO
model = load_model('cow_256.h5', custom_objects={'triplet_loss': triplet_loss})
value = 0
i = np.zeros((1, 256))
for j in glob.glob('./*.jpg'):
    print(j)
    img = image.load_img(j, target_size=(229, 229))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    i = np.vstack((i, model.predict(img_tensor)))

print(i.shape)
buf = np.zeros((5, 5))
print(buf.shape)
for u in range(1, 6):
    for v in range(u, 6):
        buf[u-1, v-1] = buf[v-1, u-1] = np.sqrt(np.sum(np.square(np.expand_dims(i[u], axis=0) - np.expand_dims(i[v], axis=0))))
print('---------------------------------')
print(buf)
