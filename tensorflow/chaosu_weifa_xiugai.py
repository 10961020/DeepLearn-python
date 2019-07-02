# !/usr/bin/python
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/14 11:19  霍金走了一周年
'''
    帮助人工筛选出一定是错误的违法证据，减轻人工筛选的工作量
    此程序开启双进程 通过管道传输预处理好的数据 将数据预处理跟模型预测分开实现，减少GPU的等待时间，加快运算速度，使用SSD算法定位车与车牌，然后加上逻辑判断
    1080ti 环境下使用 大概一秒8张图片数据的处理
'''
# !/usr/bin/python
# encoding: utf-8
# Author: zhangtong
# Time: 2019/3/14 11:19

import os
import time
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from multiprocessing import Process, Pipe


# TODO 数据预处理
def tf_pretreatment(image_path, path, result_path_list):
    image_1 = Image.open(os.path.join(path, image_path))
    try:
        if 0 < image_1.size[0] - image_1.size[1] < 500:
            img1 = image_1.crop((image_1.size[0]/2, 0, image_1.size[0], image_1.size[1]/2))
            img2 = image_1.crop((0, image_1.size[1]/2, image_1.size[0]/2, image_1.size[1]))
            tf_list, img_list = load_image_into_numpy_array([img1, img2])
        elif image_1.size[0] > image_1.size[1]:  # 左右拼接
            img1 = image_1.crop((0, 0, image_1.size[0]/2, image_1.size[1]))
            img2 = image_1.crop((image_1.size[0]/2, 0, image_1.size[0], image_1.size[1]))
            tf_list, img_list = load_image_into_numpy_array([img1, img2])
        else:  # 上下拼接
            img1 = image_1.crop((0, 0, image_1.size[0], image_1.size[1]/2))
            img2 = image_1.crop((0, image_1.size[1]/2, image_1.size[0], image_1.size[1]))
            tf_list, img_list = load_image_into_numpy_array([img1, img2])
        return [tf_list, img_list]
    except OSError:  # 可以在这里删除错误图片 我没删
        print('这个有问题的图片{}'.format(image_path))
        with open('result_chaosu.txt', 'a')as f:
            f.write('{}#1#E0015 \n'.format(image_path.split(".")[0]))
        save_img(path, image_path, "{}.jpg".format(image_path.split(".")[0]), result_path_list[1])
        return None


def load_image_into_numpy_array(image_list):
    tf_loadlist, img_loadlist = [], []
    for i in image_list:
        img1 = i.crop((0, 0, i.size[0], i.size[1]))
        img_loadlist.append(image.img_to_array(img1).astype(np.uint8))
        tf_loadlist.append(np.expand_dims(img_loadlist[-1], axis=0))
    return tf_loadlist, img_loadlist


# TODO 移动图片到指定位置
def save_img(path, image_path, img_name, result_path):
    # return None
    shutil.copy(os.path.join(path, image_path), os.path.join(result_path, img_name))


# TODO 非极大值抑制
def py_cpu_nms(dets, scores, thresh=0.5):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2 赋值
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交的面积,不重叠时面积为0

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
    return keep


# TODO 模型预测
def models_yuce(con):
    detection_graph = tf.Graph()  # 加载目标定位模型
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    c1, c2 = con
    c1.close()  # 主进程用conn1发送数据,子进程要用对应的conn2接受,所以讲conn1关闭,不关闭程序会阻塞
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with detection_graph.as_default():
        with tf.Session(config=config) as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            while True:
                try:  # 异常处理,出现异常退出
                    tf_list = c2.recv()
                    value = 0  # 0左1右
                    plate_dict = False
                    for image_np_expanded in tf_list[0]:  # 一组违法一张一张跑
                        # print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
                        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
                        # print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]
                        image2 = Image.fromarray(tf_list[1][value])

                        car_list, plate_list, carwei_list = [], [], []
                        car_list_scores, carwei_list_scores = [], []
                        for i in range(output_dict['num_detections']):  # 收集每个图片第一张车跟车牌 如果有的话
                            if output_dict['detection_scores'][i] < 0.5:
                                break
                            if output_dict['detection_classes'][i] == 2 and not plate_list:
                                plate_list.append(output_dict['detection_boxes'][i])
                            if output_dict['detection_classes'][i] == 1:
                                car_list.append(np.append(output_dict['detection_boxes'][i], np.ones(1)))
                                car_list_scores.append(output_dict['detection_scores'][i])
                            if output_dict['detection_classes'][i] == 3:
                                carwei_list.append(np.append(output_dict['detection_boxes'][i], np.zeros(1)))
                                carwei_list_scores.append(output_dict['detection_scores'][i])
                        if car_list or carwei_list:
                            # print('1 car_list', car_list)
                            # print('1 carwei_list', carwei_list)
                            if car_list and carwei_list:
                                # print(car_list_scores, carwei_list_scores)
                                car_num = np.r_[car_list, carwei_list]
                                car_scores = np.append(car_list_scores, carwei_list_scores)
                            elif not carwei_list:
                                car_num = np.array(car_list)
                                car_scores = np.array(car_list_scores)
                            else:
                                car_num = np.array(carwei_list)
                                car_scores = np.array(carwei_list_scores)
                            car_list, carwei_list = [], []
                            b = np.array([image2.size[1], image2.size[0], image2.size[1], image2.size[0], 1.], dtype=float)
                            car_num = car_num * b
                            b = py_cpu_nms(car_num, car_scores)
                            num = 0
                            # print(b)
                            for i in b:
                                if car_num[i][-1]:
                                    car_list.append(car_num[i])
                                else:
                                    carwei_list.append(car_num[i])
                                num += 1
                                if num == 2:
                                    break
                            # print('2 car_list', car_list)
                            # print('2 carwei_list', carwei_list)
                        if carwei_list and not car_list:  # 仅有车尾
                            with open('result_chaosu.txt', 'a')as f:
                                f.write('{}#1#E0607 \n'.format(tf_list[2].split(".")[0]))  # 车尾607
                            save_img(tf_list[3], tf_list[2], "{}.jpg".format(tf_list[2].split(".")[0]), tf_list[4][1])
                            plate_dict = False
                            break
                        elif not car_list:
                            with open('result_chaosu.txt', 'a')as f:
                                if not value:
                                    f.write('{}#1#E0604 \n'.format(tf_list[2].split(".")[0]))  # 车身不完整 604
                                else:
                                    f.write('{}#1#E0606 \n'.format(tf_list[2].split(".")[0]))  # 车身不完整 606
                            save_img(tf_list[3], tf_list[2], "{}.jpg".format(tf_list[2].split(".")[0]), tf_list[4][1])
                            plate_dict = False
                            break
                        elif not plate_list:
                            with open('result_chaosu.txt', 'a')as f:
                                if not value:
                                    f.write('{}#1#E0603 \n'.format(tf_list[2].split(".")[0]))  # 车身不完整 603
                                else:
                                    f.write('{}#1#E0605 \n'.format(tf_list[2].split(".")[0]))  # 车身不完整 605
                            save_img(tf_list[3], tf_list[2], "{}.jpg".format(tf_list[2].split(".")[0]), tf_list[4][1])
                            plate_dict = False
                            break
                        elif len(car_list)+len(carwei_list) == 2:
                            if car_list and carwei_list:
                                car_num = np.r_[car_list, carwei_list]
                            elif not carwei_list:
                                car_num = np.array(car_list)
                            else:
                                car_num = np.array(carwei_list)
                            car_1 = (car_num[0][2] - car_num[0][0])/2 + car_num[0][0]
                            car_2 = (car_num[1][2] - car_num[1][0])/2 + car_num[1][0]
                            if abs(car_1 - car_2) < 200:
                                with open('result_chaosu.txt', 'a')as f:
                                    f.write('{}#1#E0602 \n'.format(tf_list[2].split(".")[0]))  # 并行602
                                save_img(tf_list[3], tf_list[2], "{}.jpg".format(tf_list[2].split(".")[0]), tf_list[4][1])
                                plate_dict = False
                                break

                        plate_dict = True
                        value += 1
                    if plate_dict:
                        with open('result_chaosu.txt', 'a')as f:
                            f.write('{}#0#0 \n'.format(tf_list[2].split(".")[0]))
                        save_img(tf_list[3], tf_list[2], "{}.jpg".format(tf_list[2].split(".")[0]), tf_list[4][0])
                except EOFError:
                    break


# TODO 超速主进程
def chaosu_function(waifa_number):
    print('超速违法打开 ', waifa_number)
    test_image_paths = '/ping-data/cs'  # 原图片路径
    # test_image_paths = r'C:\Users\Administrator\Desktop\1'  # 原图片路径
    result_image_paths = '/ping-data/result/{}'.format(waifa_number)
    result_leibie_list = ['good', 'error']  # 图片分类保存的根路径
    result_path_list = []
    for i in range(len(result_leibie_list)):
        result_path_list.append(os.path.join(result_image_paths, result_leibie_list[i]))
        if not os.path.exists(result_path_list[i]):
            os.makedirs(result_path_list[i])

    conn1, conn2 = Pipe()  # 开启管道
    p = Process(target=models_yuce, args=((conn1, conn2),))  # 将管道的两个返回值以元组形式传给子进程
    p.start()
    conn2.close()  # 用conn1发送数据,conn2不用,将其关闭
    for path, dirs, files in os.walk(test_image_paths):
        for image_path in files:
            print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), os.path.join(path, image_path))
            tfrecord_list = tf_pretreatment(image_path, path, result_path_list)  # 图片预处理
            if tfrecord_list:  # 能进这说明这张不是损坏图片
                tfrecord_list.append(image_path)
                tfrecord_list.append(path)
                tfrecord_list.append(result_path_list)
                conn1.send(tfrecord_list)
        break

    conn1.close()
    p.join()

if __name__ == '__main__':
    start = time.time()
    chaosu_function('ddcs')
    end = time.time()
    print(end-start)
