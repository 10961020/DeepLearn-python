# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/4/30 11:06
import os
from keras import regularizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import numpy as np
'''
         此模型跟车牌生成器生成的相比增加了 学 警 挂 三类车牌 当然需要自己标注这样的数据集才可以识别
         没有处理新能源汽车的8位车牌 需要的话 可以在每个图片数据集不够8位的名称后增加一个缺失位 
         如'晋M12345_'  替代模型网络结构的输出层增加一个输出 7->8  修改地方代码处已指出 同理 可以增加位数不够的车牌图片 缺失位都替换成'_'
         chars列表最后一位在增加这个缺失符号'_' 同样修改位置已给出
         未使用迁移学习 所以代码需要大量的车牌数据集 (这是一个痛苦的过程)
         
         网络结构可以随意修改 如有遇到过拟合欠拟合问题
'''
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z", "学", "警", "挂"]  # --->]前边加个 ,'_'
M_strIdx = dict(zip(chars, range(len(chars))))

# 一下两个方法运行时需要注释另一个
# TODO 这个是没有做数据增强的训练集生成器
def gen_train(batch_size=32):
    processed_data = np.load('1.npy')
    training_images = processed_data[0]
    training_labels = processed_data[1]
    text_images = processed_data[4]
    text_labels = processed_data[5]
    # l_platestr, l_plateimg = [], []
    start_size = 0
    over_size = start_size+batch_size
    training_len = len(training_images)
    while True:
        if over_size > training_len:
            l_platestr = training_labels[start_size:]
            l_platestr.extend(text_labels[:over_size-training_len])
            l_plateimg = training_images[start_size:]
            l_plateimg.extend(text_images[:over_size - training_len])
            start_size = 0
            over_size = start_size + batch_size
        else:
            l_platestr = training_labels[start_size:over_size]
            l_plateimg = training_images[start_size:over_size]
        training_np = np.array(l_plateimg, dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_platestr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
        for batch in range(batch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1
        start_size = over_size
        over_size = start_size + batch_size
        yield training_np, [yy for yy in y]


# TODO 这个是做数据增强的训练集生成器
def gen_train(batch_size=32):
    processed_data = np.load('plate.npy')
    training_images = processed_data[0]
    training_labels = processed_data[1]
    training_np = np.array(training_images, dtype=np.uint8)

    train_datagen = ImageDataGenerator(
        rotation_range=5,  # 角度值，图像随机旋转的角度范围
        width_shift_range=0.1,  # 水平方向上平移的范围
        height_shift_range=0.1,  # 垂直方向上平移的范围
        fill_mode='nearest',
        zoom_range=0.1)  # 图像随机缩放的范围
    training_np = train_datagen.flow(training_np, training_labels)  # 默认取32个一次
    while True:
        a = training_np.next()
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], a[1])), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
        for batch in range(batch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1
        yield a[0], [yy for yy in y]


def gen_val(batch_size=32):
    processed_data = np.load('1.npy')
    val_images = processed_data[2]
    val_labels = processed_data[3]
    start_size = 0
    over_size = start_size+batch_size
    training_len = len(val_images)
    while True:
        if over_size > training_len:
            start_size = 0
            over_size = start_size + batch_size
        l_platestr = val_labels[start_size:over_size]
        l_plateimg = val_images[start_size:over_size]
        val_np = np.array(l_plateimg, dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_platestr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
        for batch in range(batch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1
        start_size = over_size
        over_size = start_size + batch_size
        yield val_np, [yy for yy in y]

if __name__ == '__main__':
    if os.path.exists('chepai_best.h5'):
        model = load_model('chepai_best.h5')
    else:
        adam = Adam(lr=0.001)
        input_tensor = Input((42, 132, 3))
        x = input_tensor
        for i in range(3):
            x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
            x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
            x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.25)(x)
        x = [Dense(256, kernel_regularizer=regularizers.l2(0.001),  activation='relu')(x) for i in range(7)]
        n_class = len(chars)
        x = [Dense(n_class, activation='softmax', name='c%d' % (i+1))(x[i]) for i in range(7)]
        model = Model(inputs=input_tensor, outputs=x)
        print(model.summary())
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

    best_model = ModelCheckpoint("chepai_best.h5", monitor='val_loss', verbose=0, save_best_only=True)

    model.fit_generator(gen_train(32), steps_per_epoch=2000, epochs=40000,
                        validation_data=gen_val(32), validation_steps=17,
                        callbacks=[best_model])
