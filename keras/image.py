# !/usr/bin/python  
# encoding: utf-8
# author: zhangtong

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    # rescale=1./255,             # 图像乘以1./255 缩放到0-1之间
    rotation_range=60,          # 角度值，图像随机旋转的角度范围
    width_shift_range=0.3,      # 水平方向上平移的范围
    height_shift_range=0.1,     # 垂直方向上平移的范围
    shear_range=0.2,            # 随机错切变换的角度
    zoom_range=0.1,             # 图像随机缩放的范围
    horizontal_flip=True,
    fill_mode='nearest')       # 随机将一半图像水平翻转
img_path = 'D:/1/project/tensorflow_NO1/wusun/train/Unsuspended/0610500200952810903.jpg'
img = image.load_img(img_path, target_size=(400, 400))
x = image.img_to_array(img)
x = x.reshape((1,)+x.shape)
i = 0
for batch in train_datagen.flow(x, batch_size=1):
    img = image.array_to_img(batch[0])
    img.save('./{}.jpg'.format(i))
    i += 1
    if i >= 5:
        break
