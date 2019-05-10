# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/4/26 14:44

import os
import time
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from multiprocessing import Process, Pipe
from object_detection.utils import ops as utils_ops
import label_map_util
import visualization_utils as vis_util


# TODO 数据预处理
def tf_pretreatment(image_path, files, path):
    try:
        image_1 = Image.open(os.path.join(path, image_path))
        tf_list, img_list = load_image_into_numpy_array([image_1])
        return [tf_list, img_list]
    except OSError:  # 可以在这里删除错误图片 我没删
        print('这个有问题的图片{}'.format(image_path))
        return None
    except TypeError:  # 可以在这里删除错误图片 我没删
        print('这个有问题的图片1{}'.format(image_path))
        return None


def load_image_into_numpy_array(image_list):
    tf_loadlist, img_loadlist = [], []
    for i in image_list:
        # img1 = i.crop((0, i.size[1] * 0.2, i.size[0], i.size[1]))
        img_loadlist.append(image.img_to_array(i).astype(np.uint8))
        tf_loadlist.append(np.expand_dims(img_loadlist[-1], axis=0))
    return tf_loadlist, img_loadlist


# TODO 移动图片到指定位置
def save_img(path, image_path, files, result_path):
    shutil.move(os.path.join(path, image_path), os.path.join(result_path, image_path))
    linshi_file = image_path.split('_')
    linshi_file[6] = '2'
    linshi_file = '_'.join(linshi_file)
    if linshi_file in files:
        shutil.move(os.path.join(path, linshi_file), os.path.join(result_path, linshi_file))


# TODO 模型预测
def models_yuce(con):
    num_classes = 90
    path_to_labels = 'D:/1/exe/mask_rcnn/mask_rcnn_test-master/training/mscoco_label_map.pbtxt'
    detection_graph = tf.Graph()  # 加载目标定位模型
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('D:/1/exe/mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_to_labels)  # 加载目标定位类别编号
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

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
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            while True:
                try:  # 异常处理,出现异常退出
                    tf_list = c2.recv()
                    value = 0  # 0左1右
                    for image_np_expanded in tf_list[0]:  # 一组违法一张一张跑
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image_np_expanded.shape[1], image_np_expanded.shape[2])
                        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '开始')
                        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
                        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '结束')
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]

                        vis_util.visualize_boxes_and_labels_on_image_array(
                            tf_list[1][value],
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        image2 = Image.fromarray(tf_list[1][value])
                        image2.save('result/{}_{}{}'.format(os.path.splitext(tf_list[2])[0], value, os.path.splitext(tf_list[2])[1]))
                        value += 1
                except EOFError:  # 说明素有数据已经全部接受,进程会抛出异常
                    break


# TODO 超速主进程
def chaosu_function(waifa_number):
    print('超速违法打开 ', waifa_number)
    with open('result_chaosu.csv', 'w'):
        pass
    test_image_paths = 'C:/Users/Administrator/Desktop/校验集/{}'.format(waifa_number)  # 原图片路径
    result_image_paths = 'D:/1/project/wuxi/result/{}'.format(waifa_number)  # 图片保存路径
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
            if image_path.split('_')[6] != '1':  # 碰到组合图片不是第一张图片直接跳过
                continue
            print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), os.path.join(path, image_path))
            tfrecord_list = tf_pretreatment(image_path, files, path)  # 图片预处理
            if tfrecord_list:  # 能进这说明这张不是损坏图片
                tfrecord_list.append(image_path)
                tfrecord_list.append(path)
                tfrecord_list.append(files)
                tfrecord_list.append(result_path_list)
                conn1.send(tfrecord_list)
    conn1.close()
    p.join()

if __name__ == '__main__':
    chaosu_function('ddcs')
