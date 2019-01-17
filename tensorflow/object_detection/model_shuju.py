# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/1/11 10:53

import os
import numpy as np
import sys
import tensorflow as tf
# import matplotlib.pyplot as plt
from PIL import Image

# 将目标目录导入进来，这样才能执行下边两句导入命令
# label_map_util,visualization_utils文件路径自行寻找 简单粗暴点 进入research目录全局搜这俩模块名
sys.path.insert(0, "/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/object_detection/")
from utils import label_map_util
from utils import visualization_utils as vis_util
import xml.etree.ElementTree as ET


def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# frozen_inference_graph.pb中保存的网络的结构跟数据
# own_label_map.pbtxt中保存index到类别名的映射 需要通过pbtxt具体对应的类别是什么
# 第三个全局是自训练的目录 路径名最后一层必须是 “img” 因为我懒得加这个在xml里 如果不是 打的标签不能用
# 第四个保存的xml路径
# oragn也是重要的参数 保存的自训练模版的文件 可以看到第二行的数据就是 img
PATH_TO_FROZEN_GRAPH = '/data/Deeplearn/wusun/car_dingwei/export/frozen_inference_graph.pb'
PATH_TO_LABELS = '/data/Deeplearn/wusun/car_dingwei/own_label_map.pbtxt'
TEST_IMAGE_PATHS = '/data/Deeplearn/wusun/car_dingwei/img/'
TEST_CAR_PATHS = '/data/Deeplearn/wusun/car_dingwei/img/car/'
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
value = 1
x = 1
with detection_graph.as_default():
    with tf.Session() as sess:
        for path, dirs, files in os.walk(TEST_IMAGE_PATHS):
            for image_path in files:
                image_1 = Image.open(path+image_path)
                # 图片转换为numpy的形式
                image_np = load_image_into_numpy_array(image_1)
                # 图片扩展一维，最后进入神经网络的图片格式应该为[1, ?, ?, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                        'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                # plt.figure(figsize=(12, 8))
                # plt.imshow(image_np)
                # plt.show()
                # print("output_dict['detection_boxes']: ", output_dict['detection_boxes'])
                # print("output_dict['detection_classes']: ", output_dict['detection_classes'])
                # print("output_dict['detection_scores']: ", output_dict['detection_scores'])
                # print("output_dict['num_detections']: ", output_dict['num_detections'])
                print(x)
                x += 1
                image2 = Image.fromarray(image_np)
                # image2.save('{}.jpg'.format(value))
                # value += 1
                for i in range(output_dict['num_detections']):
                    if output_dict['detection_scores'][i] < 0.8:
                        break
                    img = image2.crop((int(image2.size[0] * output_dict['detection_boxes'][i][1]),
                                       int(image2.size[1] * output_dict['detection_boxes'][i][0]),
                                       int(image2.size[0] * output_dict['detection_boxes'][i][3]),
                                       int(image2.size[1] * output_dict['detection_boxes'][i][2])))
                    img.save(TEST_CAR_PATHS+'car_{}.jpg'.format(value))
                    value += 1
            break

