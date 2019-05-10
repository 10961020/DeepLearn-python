# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/5 15:53
'''
         程序使用到的数据集是车牌生成器随机生成的数据
         genplate()里边用到的都是相对路径 所以可能会报错 就是因为没有找到对应文件 把 当前文件跟genplate.py同级存放就可以 
         但是同级存放需要from plate.genplate import * 改为 import genplate
         参考https://cloud.tencent.com/developer/article/1005199
'''
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from plate.genplate import *

np.random.seed(5)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"]

M_strIdx = dict(zip(chars, range(len(chars))))

n_generate = 100
rows = 20
cols = int(n_generate/rows)

# l_plateStr, l_plateImg = G.genBatch(100, 2, range(31, 65), "./plate", (272, 72))
#
# l_out = []
# for i in range(rows):
#     l_tmp = []
#     for j in range(cols):
#         l_tmp.append(l_plateImg[i*cols+j])
#
#     l_out.append(np.hstack(l_tmp))
#
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.imshow(np.vstack(l_out), aspect="auto")
# plt.show()


def gen(batch_size=32):
    G = GenPlate("./plate/font/platech.ttf", './plate/font/platechar.ttf', "./plate/NoPlates")
    while True:
        l_platestr, l_plateimg = G.genBatch(batch_size, 2, range(31, 65), "./plate", (272, 72))
        X = np.array(l_plateimg, dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_platestr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
        for batch in range(batch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1
        yield X, [yy for yy in y]

adam = Adam(lr=0.001)

input_tensor = Input((72, 272, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)

n_class = len(chars)
x = [Dense(n_class, activation='softmax', name='c%d' % (i+1))(x) for i in range(7)]
model = Model(inputs=input_tensor, outputs=x)
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

best_model = ModelCheckpoint("chepai_best.h5", monitor='val_loss', verbose=0, save_best_only=True)

model.fit_generator(gen(32), steps_per_epoch=2000, epochs=5,
                    validation_data=gen(32), validation_steps=1280,
                    callbacks=[best_model])
