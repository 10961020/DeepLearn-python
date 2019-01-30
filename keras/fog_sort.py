#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     :2018/12/20 9:14
# @Author   :tong.z

import os
import cv2
import time
import shutil
import threading
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
'''
    一个可以初版上线的团雾检测统计以及收集更多的公路数据的程序

    需要修改的有PATH_TO_VIDEO、TECT_IMAGE_PATHS变量      <----重点!!!
    每次迭代生成的txt说明文件在程序根目录下 文件名为当时生成文件的时间戳
    文件说明 缩写：文件名     | 缩写如下 文件名在各个缩写的文件中
    两个可能有或没雾的文件夹 因为图片的预测结果在 0.9-0.1 之间 不是确定是不是有雾的分类文件夹
    fog     no_fog     maybe_fog    maybe_no_fog         
    有雾     没雾       可能有雾      可能没雾

    视频每隔50帧截取一张 先截取最新的视频文件 如有不够截取图片张数 将读取之前的视频文件
    50帧 可以根据实际情况进行修改
'''
sb_id = {}
IMG_NUMBER = 3  # 一次保存从视频中截取几张图片
PATH_TO_Classification = 'fog_9770.h5'
PATH_TO_VIDEO = '/data/TRAS/video/video/'  # 视频根目录
TECT_IMAGE_PATHS = '/data/TRAS/fog/'  # 程序根目录


def main():
    with open(os.path.join(TECT_IMAGE_PATHS, '{}.txt'.format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))), 'w') as f:
        for files in sb_id:
            for jpg_path in sb_id[files]:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '打开 {} 图片'.format(jpg_path))
                img = image.load_img(os.path.join(TECT_IMAGE_PATHS, jpg_path), target_size=(200, 200))
                img_tensor = image.img_to_array(img)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                img_tensor /= 255
                i = model.predict(img_tensor)
                # print(jpg_path + ': ' + str(1 if i[0][0] < 0.5 else 0))
                if i[0][0] < 0.1:
                    value = 0
                elif i[0][0] < 0.5:
                    value = 2
                elif i[0][0] < 0.9:
                    value = 3
                else:
                    value = 1
                shutil.move(os.path.join(TECT_IMAGE_PATHS, jpg_path), os.path.join(path_list[value], jpg_path))
                f.write('fog: ' if i[0][0] < 0.5 else 'no_fog: ' + os.path.join(path_list[value], jpg_path)+'\n')


# TODO 从视频中截取照片
def video_to_img():
    global sb_id
    global IMG_NUMBER
    sb_id = {}
    IMG_NUMBER = 3
    for path, dirs, files in os.walk(PATH_TO_VIDEO):
        files = [x for x in files if os.path.splitext(x)[1] == '.dat']  # 筛选出只有该扩展名的视频文件
        files = sorted(files, reverse=True, key=lambda x: os.path.getmtime(os.path.join(path, x)))  # 根据修改时间排序 最新生成的先读
        for image_path in files:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '读取视频: ', image_path)
            cap = cv2.VideoCapture(os.path.join(path, image_path))
            video_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_num += 1
                if not video_num % 50:
                    cv2.imwrite(os.path.join(TECT_IMAGE_PATHS, os.path.splitext(image_path)[0] + '{}.jpg'.format(IMG_NUMBER)), frame)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                          '生成的图片: ' + os.path.splitext(image_path)[0] + '{}.jpg'.format(IMG_NUMBER))
                    if image_path[:image_path.find('_')] not in sb_id:
                        sb_id[image_path[:image_path.find('_')]] = []
                    sb_id[image_path[:image_path.find('_')]].append(os.path.splitext(image_path)[0] + '{}.jpg'.format(IMG_NUMBER))
                    IMG_NUMBER -= 1
                    if not IMG_NUMBER:
                        cap.release()
                        return
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()


# TODO 定时任务每隔十分钟开始执行一次
def timed_task():
    timer = threading.Timer(600, timed_task)  # 十分钟触发一次
    timer.start()
    video_to_img()
    if IMG_NUMBER < 3:
        main()
    else:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '当前时间没有视频')


if __name__ == '__main__':
    # 打开车牌分类模型
    model = load_model(PATH_TO_Classification)
    leibie_list = ['fog/', 'no_fog/', 'maybe_fog/', 'maybe_no_fog/']
    path_list = []
    for j in range(len(leibie_list)):
        path_list.append(os.path.join(TECT_IMAGE_PATHS, leibie_list[j]))
        if not os.path.exists(path_list[j]):
            os.makedirs(path_list[j])

    timed_task()
