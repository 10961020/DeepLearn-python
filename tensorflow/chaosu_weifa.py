# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/14 11:19

import os
import sys
import time
import shutil
import numpy as np
import label_map_util
from PIL import Image
import tensorflow as tf
# import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from multiprocessing import Process, Pipe

# frozen_inference_graph.pb中保存的网络的结构跟数据
# own_label_map.pbtxt中保存index到类别名的映射 需要通过pbtxt具体对应的类别是什么
# PATH_TO_Classification 车牌识别模型
# 第四个全局是自训练的目录 路径名最后一层必须是 “img” 因为我懒得加这个在xml里 如果不是 打的标签不能用
# 第五个图片分类的根路径
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PATH_TO_Classification = 'chepai_best.h5'
PATH_TO_LABELS = 'own_label_map.pbtxt'
TEST_IMAGE_PATHS = 'C:/Users/Administrator/Desktop/校验集'
RESULT_IMAGE_PATHS = 'D:/1/project/wuxi'
NUM_CLASSES = 1
plate_dict = {}  # 格式内容 {0(左图):[车牌图片],1(右图):[车牌图片]}
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"]

result_leibie_list = ['good', 'error']  # 图片分类保存的根路径
result_path_list = []
for i in range(len(result_leibie_list)):
    result_path_list.append(os.path.join(RESULT_IMAGE_PATHS, result_leibie_list[i]))
    if not os.path.exists(result_path_list[i]):
        os.makedirs(result_path_list[i])


def tf_pretreatment(image_path, files, path):
    tf_list = []
    img_list = []
    file = image_path.split('_')
    file[6] = '2'
    file = '_'.join(file)
    image_1 = Image.open(os.path.join(path, image_path))
    try:
        if file in files:  # 多张一组
            image_2 = Image.open(os.path.join(path, file))
            image_1 = image_1.crop((0, image_1.size[1]*0.2, image_1.size[0], image_1.size[1]))
            image_2 = image_2.crop((0, image_2.size[1]*0.2, image_2.size[0], image_2.size[1]))
            tf_list.append(load_image_into_numpy_array(image_1))
            tf_list.append(load_image_into_numpy_array(image_2))
            img_list.append(np.array(image_1.getdata()).reshape((image_1.size[1], image_1.size[0], 3)).astype(np.uint8))
            img_list.append(np.array(image_2.getdata()).reshape((image_2.size[1], image_2.size[0], 3)).astype(np.uint8))
        else:
            if image_1.size[0] > image_1.size[1]:  # 左右拼接
                img1 = image_1.crop((0, 0, image_1.size[0]/2, image_1.size[1]))
                img2 = image_1.crop((image_1.size[0]/2, 0, image_1.size[0], image_1.size[1]))
                img1 = img1.crop((0, img1.size[1] * 0.2, img1.size[0], img1.size[1]))
                img2 = img2.crop((0, img2.size[1] * 0.2, img2.size[0], img2.size[1]))
                tf_list.append(load_image_into_numpy_array(img1))
                tf_list.append(load_image_into_numpy_array(img2))
                img_list.append(np.array(img1.getdata()).reshape((img1.size[1], img1.size[0], 3)).astype(np.uint8))
                img_list.append(np.array(img2.getdata()).reshape((img2.size[1], img2.size[0], 3)).astype(np.uint8))
            else:  # 上下拼接
                img1 = image_1.crop((0, 0, image_1.size[0], image_1.size[1]/2))
                img2 = image_1.crop((0, image_1.size[1]/2, image_1.size[0], image_1.size[1]))
                img1 = img1.crop((0, img1.size[1] * 0.2, img1.size[0], img1.size[1]))
                img2 = img2.crop((0, img2.size[1] * 0.2, img2.size[0], img2.size[1]))
                tf_list.append(load_image_into_numpy_array(img1))
                tf_list.append(load_image_into_numpy_array(img2))
                img_list.append(np.array(img1.getdata()).reshape((img1.size[1], img1.size[0], 3)).astype(np.uint8))
                img_list.append(np.array(img2.getdata()).reshape((img2.size[1], img2.size[0], 3)).astype(np.uint8))
        return [tf_list, img_list]
    except OSError:  # 可以在这里删除错误图片 我没删
        print('这个有问题的图片{}'.format(image_path))
        return None


def load_image_into_numpy_array(image_1):
    image_np = np.array(image_1.getdata()).reshape((image_1.size[1], image_1.size[0], 3)).astype(np.uint8)
    return np.expand_dims(image_np, axis=0)


def save_img(path, image_path, files, num):
    shutil.move(os.path.join(path, image_path), os.path.join(result_path_list[num], image_path))
    linshi_file = image_path.split('_')
    linshi_file[6] = '2'
    linshi_file = '_'.join(linshi_file)
    if linshi_file in files:
        shutil.move(os.path.join(path, linshi_file), os.path.join(result_path_list[num], linshi_file))


def models_yuce(con):
    detection_graph = tf.Graph()  # 加载目标定位模型
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)  # 加载目标定位类别编号
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # model = load_model(PATH_TO_Classification)

    c1, c2 = con
    c1.close()  # 主进程用conn1发送数据,子进程要用对应的conn2接受,所以讲conn1关闭,不关闭程序会阻塞
    while True:
        try:  # 异常处理,出现异常退出
            tf_list = c2.recv()
            value = 0  # 0左1右
            plate_dict.clear()
            with detection_graph.as_default():
                with tf.Session() as sess:
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                    for image_np_expanded in tf_list[0]:  # 一组违法一张一张跑
                        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
                        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
                        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]

                        image2 = Image.fromarray(tf_list[1][value])
                        car_list, plate_list = [], []
                        for i in range(output_dict['num_detections']):  # 收集每个图片第一张车跟车牌 如果有的话
                            if output_dict['detection_scores'][i] < 0.5 or (len(car_list) == 2 and plate_list):
                                break
                            if output_dict['detection_classes'][i] == 2 and not plate_list:
                                plate_list.append(output_dict['detection_boxes'][i])
                            if output_dict['detection_classes'][i] == 1 and len(car_list) < 2:
                                car_list.append(output_dict['detection_boxes'][i])
                        if len(car_list) == 2 and int(image2.size[1] * car_list[1][2]) > int(image2.size[1] * car_list[0][0]):
                            with open('result.csv', 'a')as f:
                                f.write(tf_list[2] + ' 0 1 {}2\n'.format(value))  # 并行 2
                            plate_dict.clear()
                            save_img(tf_list[3], tf_list[2], tf_list[4], 1)
                            break
                        if plate_list:  # 如果有车牌 肯定是有车
                            img = image2.crop((int(image2.size[0] * plate_list[0][1]),
                                               int(image2.size[1] * plate_list[0][0]),
                                               int(image2.size[0] * plate_list[0][3]),
                                               int(image2.size[1] * plate_list[0][2])))
                            plate_dict[value] = img
                        elif car_list:  # 如果只有车牌 判断车牌底的位置跟图片底部位置关系
                            if image2.size[1] - int(image2.size[1] * car_list[0][2]) < 90:
                                with open('result.csv', 'a')as f:
                                    f.write(tf_list[2] + ' 0 1 {}1\n'.format(value))  # 车身不完整 1
                                plate_dict.clear()
                                save_img(tf_list[3], tf_list[2], tf_list[4], 1)
                                break
                            elif value == 1 and not plate_dict:  # 两辆车都没找到车牌 但是车位置明显完整
                                with open('result.csv', 'a')as f:
                                    f.write(tf_list[2] + ' 1 0\n')  # 先默认车牌都对
                                save_img(tf_list[3], tf_list[2], tf_list[4], 0)
                        else:  # 能到这的图片都没找到 那就是没车喽
                            with open('result.csv', 'a')as f:
                                f.write(tf_list[2] + ' 0 1 {}0\n'.format(value))  # 没找到车 0
                            plate_dict.clear()
                            save_img(tf_list[3], tf_list[2], tf_list[4], 1)
                            break
                        value += 1

            if plate_dict:
                with open('result.csv', 'a')as f:
                    f.write(tf_list[2] + ' 1 0\n')  # 先默认车牌都对
                save_img(tf_list[3], tf_list[2], tf_list[4], 0)
                # print(tf_list[2].split('_')[3])
                # for plate_key, plate_value in plate_dict.items():
                #     img_tensor = image.img_to_array(plate_value.resize((272, 72)))
                #     plate_value.resize((272, 72)).save('{}.jpg'.format(plate_key))
                #     img_tensor = np.expand_dims(img_tensor, axis=0)
                #     i = model.predict(img_tensor)
                #     for j in range(7):
                #         print(chars[int(np.where(i[j] == np.max(i[j]))[1])], end='')
                #     print()
        except EOFError:  # 说明素有数据已经全部接受,进程会抛出异常
            break

if __name__ == '__main__':
    conn1, conn2 = Pipe()  # 开启管道
    p = Process(target=models_yuce, args=((conn1, conn2),))  # 将管道的两个返回值以元组形式传给子进程
    p.start()
    conn2.close()  # 用conn1发送数据,conn2不用,将其关闭
    for path, dirs, files in os.walk(TEST_IMAGE_PATHS):
        for image_path in files:
            if image_path.split('_')[6] != '1':  # 碰到组合图片不是第一张图片直接跳过
                continue
            print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), os.path.join(path, image_path))
            tfrecord_list = tf_pretreatment(image_path, files, path)  # 图片预处理
            print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '预处理时间')
            if tfrecord_list:  # 能进这说明这张不是损坏图片
                tfrecord_list.append(image_path)
                tfrecord_list.append(path)
                tfrecord_list.append(files)
                conn1.send(tfrecord_list)


