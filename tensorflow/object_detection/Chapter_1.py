# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/1/7 10:33
'''
    目标检测打标签是一件特别痛苦的事情，此模块可以使用一个小型的标签数据训练好的模型 帮助你进行大规模的打标签的工作
    首先你需要有一个小型的训练模型帮助你做打标签 这个模型自己训练喽 xml使用的模版我已给出 pbtxt如果你有小型模型的话 这个文件你也应该有的
    代码中注释的部分都可以修改 我已给出作用
'''
import os
import time
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import xml.etree.ElementTree as ET
from keras.preprocessing import image
from multiprocessing import Process, Pipe

PATH_TO_FROZEN_GRAPH = 'rilang_fenkai.pb'  # 模型
TEST_IMAGE_PATHS = 'C:/Users/Administrator/Desktop/good'  # 需要标注的源数据
RESULT_IMAGE_PATHS = 'D:/1/project/wuxi/img'  # 经过标注的图片
TEST_XML_PATHS = 'D:/1/project/wuxi/xml'  # 经过标注的xml
class_num = [' ', 'car', 'person']

if not os.path.exists(RESULT_IMAGE_PATHS):
    os.makedirs(RESULT_IMAGE_PATHS)
if not os.path.exists(TEST_XML_PATHS):
    os.makedirs(TEST_XML_PATHS)


def file_split(image_path):
    file2 = image_path.split('_')
    file2[6] = '2'
    file2 = '_'.join(file2)
    file3 = image_path.split('_')
    file3[6] = '3'
    file3 = '_'.join(file3)
    return file2, file3


def tf_pretreatment(image_path, files, path):
    tf_list, img_list = [], []
    file2, file3 = file_split(image_path)
    try:
        if file2 in files and file3 in files:  # 多张一组 只看2，3两张
            for i in [file2, file3]:
                image_2 = Image.open(os.path.join(path, i))
                image_2 = image_2.crop((0, 0, image_2.size[0], image_2.size[1]*0.8))
                img_list.append(image.img_to_array(image_2).astype(np.uint8))
                tf_list.append(np.expand_dims(img_list[-1], axis=0))
        else:  # 4*4 只看2，3张
            image_1 = Image.open(os.path.join(path, image_path))
            img_1, img_find = [], []
            height, weight = image_1.size
            if height % 2:
                image_1 = image_1.resize((height-1, weight))
            if weight % 2:
                image_1 = image_1.resize((height, weight-1))
            for i in [0, 0.5]:
                for j in [0, 0.5]:
                    img_linshi = image_1.crop((image_1.size[0]*j, image_1.size[1]*i, image_1.size[0]*j+image_1.size[0]/2, image_1.size[1]*i+image_1.size[1]/2))
                    img_1.append(img_linshi.crop((0, 0, img_linshi.size[0], img_linshi.size[1]*0.8)))
                    big_image = image_1.crop((image_1.size[0]*j, image_1.size[1]*i, image_1.size[0]*j+image_1.size[0]/5, image_1.size[1]*i+image_1.size[1]/5))
                    img_find.append(np.array(big_image))
            del img_1[find_difference(img_find)]
            del img_1[0]
            for i in img_1:
                img_list.append(image.img_to_array(i).astype(np.uint8))
                tf_list.append(np.expand_dims(img_list[-1], axis=0))
        return [tf_list, img_list]
    except OSError:  # 可以在这里删除错误图片 我没删
        print('这个有问题的图片{}'.format(image_path))
        return None


# 找特写图
def find_difference(image_list, threshold=10):
    image_num = len(image_list)
    mask = np.zeros((image_num, image_num))
    for i in range(image_num):
        for j in range(i+1, image_num):
            mask[i, j] = mask[j, i] = np.sum((np.abs(image_list[i] - image_list[j])) < threshold)
    mask = np.sum(mask, axis=1)
    return np.argmin(mask)


def models_yuce(con):
    detection_graph = tf.Graph()  # 加载目标定位模型
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    c1, c2 = con
    c1.close()  # 主进程用conn1发送数据,子进程要用对应的conn2接受,所以讲conn1关闭,不关闭程序会阻塞
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
            while True:
                try:  # 异常处理,出现异常退出
                    tf_list = c2.recv()
                    value = 0  # 0左1右
                    for image_np_expanded in tf_list[0]:  # 一组违法一张一张跑
                        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]

                        image2 = Image.fromarray(tf_list[1][value])
                        image2.save(os.path.join(RESULT_IMAGE_PATHS, '{}_{}{}'.format(os.path.splitext(tf_list[2])[0], value, os.path.splitext(tf_list[2])[1])))
                        organ = "./xiugai.xml"
                        num_output = 0
                        for i in range(output_dict['num_detections']):
                            tree = ET.ElementTree()
                            tree.parse(organ)
                            if output_dict['detection_scores'][i] < 0.5:
                                break
                            if output_dict['detection_classes'][i] != 1 and output_dict['detection_classes'][i] != 2:
                                continue
                            if num_output == 0:
                                num_output += 1
                                tree.find("filename").text = tf_list[2]
                                tree.find("path").text = os.path.join(RESULT_IMAGE_PATHS, '{}_{}{}'.format(os.path.splitext(tf_list[2])[0], value, os.path.splitext(tf_list[2])[1]))
                                tree.find("size/width").text = str(image2.size[0])
                                tree.find("size/height").text = str(image2.size[1])
                            root = tree.getroot()
                            firstNode = ET.Element("object")
                            ET.SubElement(firstNode, "name").text = class_num[output_dict['detection_classes'][i]]
                            ET.SubElement(firstNode, "pose").text = 'Unspecified'
                            ET.SubElement(firstNode, "truncated").text = '0'
                            ET.SubElement(firstNode, "difficult").text = '0'
                            twoNode = ET.SubElement(firstNode, "bndbox")
                            ET.SubElement(twoNode, "xmin").text = str(int(image2.size[0] * output_dict['detection_boxes'][i][1]))
                            ET.SubElement(twoNode, "ymin").text = str(int(image2.size[1] * output_dict['detection_boxes'][i][0]))
                            ET.SubElement(twoNode, "xmax").text = str(int(image2.size[0] * output_dict['detection_boxes'][i][3]))
                            ET.SubElement(twoNode, "ymax").text = str(int(image2.size[1] * output_dict['detection_boxes'][i][2]))
                            root.append(firstNode)
                            organ = os.path.join(TEST_XML_PATHS, '{}_{}'.format(os.path.splitext(tf_list[2])[0], value) + '.xml')
                            tree.write(organ)
                        print(organ)
                        value += 1
                except EOFError:
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
            if tfrecord_list:  # 能进这说明这张不是损坏图片
                tfrecord_list.append(image_path)
                tfrecord_list.append(path)
                tfrecord_list.append(files)
                conn1.send(tfrecord_list)
        break
    conn1.close()
    p.join()
