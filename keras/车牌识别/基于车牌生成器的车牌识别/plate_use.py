# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/2/27 9:11
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
from keras.preprocessing import image
from keras.models import load_model
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"]
list1 = [ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.EDGE_ENHANCE,
         ImageFilter.EDGE_ENHANCE_MORE, ImageFilter.EMBOSS, ImageFilter.FIND_EDGES,
         ImageFilter.SMOOTH, ImageFilter.SMOOTH_MORE, ImageFilter.SHARPEN]
PATH_TO_Classification = 'chepai_best.h5'  #保存的车牌模型
# 打开车牌分类模型
model = load_model(PATH_TO_Classification)
for path, dirs, files in os.walk(r'D:\1\project\wuxi\result\plate'):
    for file in files:
        img = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img.resize((42, 132, 3))
        # blur = cv2.GaussianBlur(img, (7, 7), 0)
        # canny = cv2.Canny(blur, 50, 150)
        # canny.resize((72, 272, 1))
        # image_1 = Image.open(os.path.join(path, file))
        # print(file[file.find('_')+1:file.find('.')])  # carpai
        # print(os.path.splitext(file)[0].split('_')[3])  # 现实

        # pil尝试 边缘增强+高斯模糊
        # img_tensor = image.img_to_array(image_1.resize((272, 72)).filter(ImageFilter.EDGE_ENHANCE).filter(ImageFilter.BLUR))

        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        i = model.predict(img_tensor)
        for j in range(7):
            print(chars[int(np.where(i[j] == np.max(i[j]))[1])], end='')
        print()
    break
