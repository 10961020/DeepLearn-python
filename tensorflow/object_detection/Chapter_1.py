# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/1/7 10:33
import os
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
'''
    目标检测打标签是一件特么痛苦的事情，此模块可以使用一个小型的标签数据训练好的模型 帮助你进行大规模的打标签的工作
    首先你需要有一个小型的训练模型帮助你做打标签 这个模型自己训练喽 xml使用的模版我已给出 pbtxt如果你有小型模型的话 这个文件你也应该有的
    代码中注释的部分都可以修改 我已给出作用
'''
# 将目标目录导入进来，这样才能执行下边两句导入命令
# label_map_util,visualization_utils文件路径自行寻找 简单粗暴点 进入research目录全局搜这俩模块名
sys.path.insert(0, "D:/1/python3/Lib/site-packages/tensorflow/models/research/build/lib/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util
import xml.etree.ElementTree as ET


def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# frozen_inference_graph.pb中保存的网络的结构跟数据
# own_label_map.pbtxt中保存index到类别名的映射 需要通过pbtxt具体对应的类别是什么
# 第三个全局是自训练的目录 路径名最后一层必须是 “img” 因为我懒得加这个在xml里 如果不是 打的标签好像不能用 日常迷信一下
# 第四个保存的xml路径
# oragn也是重要的参数 保存的自训练模版的文件 可以看到第二行的数据就是 img
PATH_TO_FROZEN_GRAPH = 'C:/Users/Administrator/Desktop/car_position/frozen_inference_graph.pb'
PATH_TO_LABELS = 'C:/Users/Administrator/Desktop/car_position//own_label_map.pbtxt'
TEST_IMAGE_PATHS = 'C:/Users/Administrator/Desktop/img/'
TEST_XML_PATHS = 'C:/Users/Administrator/Desktop/img/xml/'
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

with detection_graph.as_default():
    with tf.Session() as sess:
        tree = ET.ElementTree()
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

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                # plt.figure(figsize=IMAGE_SIZE)
                # plt.imshow(image_np)
                # plt.show()
                # print("output_dict['detection_boxes']: ", output_dict['detection_boxes'])
                # print("output_dict['detection_classes']: ", output_dict['detection_classes'])
                # print("output_dict['detection_scores']: ", output_dict['detection_scores'])
                # print("output_dict['num_detections']: ", output_dict['num_detections'])

                image2 = Image.fromarray(image_np)
                image2.save('1.jpg')  # 需要保存生成的图片的话  你给他个路径 推荐使用新的文件路径 名字就可以跟使用的文件名一样啦
                organ = "./xiugai.xml"  # 这个是我给出的xml模型路径 跟程序放一块就好了不用修改
                for i in range(output_dict['num_detections']):
                    
                    tree.parse(organ)
                    if output_dict['detection_scores'][i] < 0.8: # 保留多大得分的选框 由你决定
                        break
                    if i == 0:  # 如果图片中就一个目标
                        tree.find("filename").text = image_path
                        tree.find("path").text = path+image_path
                        tree.find("size/width").text = str(image2.size[0])
                        tree.find("size/height").text = str(image2.size[1])
                        tree.find("object/bndbox/xmin").text = str(int(image2.size[0]*output_dict['detection_boxes'][i][1]))
                        tree.find("object/bndbox/ymin").text = str(int(image2.size[1]*output_dict['detection_boxes'][i][0]))
                        tree.find("object/bndbox/xmax").text = str(int(image2.size[0]*output_dict['detection_boxes'][i][3]))
                        tree.find("object/bndbox/ymax").text = str(int(image2.size[1]*output_dict['detection_boxes'][i][2]))
                        organ = TEST_XML_PATHS + os.path.splitext(image_path)[0] + '.xml'
                    else:  # 在这个图片里发现多个目标 追加object
                        root = tree.getroot()
                        firstNode = ET.Element("object")
                        ET.SubElement(firstNode, "name").text = 'car'
                        ET.SubElement(firstNode, "pose").text = 'Unspecified'
                        ET.SubElement(firstNode, "truncated").text = '0'
                        ET.SubElement(firstNode, "difficult").text = '0'
                        twoNode = ET.SubElement(firstNode, "bndbox")
                        ET.SubElement(twoNode, "xmin").text = str(int(image2.size[0]*output_dict['detection_boxes'][i][1]))
                        ET.SubElement(twoNode, "ymin").text = str(int(image2.size[1]*output_dict['detection_boxes'][i][0]))
                        ET.SubElement(twoNode, "xmax").text = str(int(image2.size[0]*output_dict['detection_boxes'][i][3]))
                        ET.SubElement(twoNode, "ymax").text = str(int(image2.size[1]*output_dict['detection_boxes'][i][2]))
                        root.append(firstNode)  # 追加到xml文件 如果有多的目标框
                    tree.write(organ)
            break

