# !/usr/bin/python  
# -*- coding:utf-8 -*-
# @Author   :tong.z
# @Time     :2019/1/11 10:53

import os
import cv2
import time
import shutil
import threading
import numpy as np
import label_map_util
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
'''
   一个可以初版上线的车辆定位并且根据污损分类统计以及更丰富的车辆数据的代码
    
    需要修改的有PATH_TO_VIDEO、TEST_IMAGE_PATHS变量      <----重点!!!
    生成图片的路径下有原图备份的文件夹 yt/ 其他几个为了数据收集用
    每次迭代生成的txt说明文件在程序根目录下 文件名为当时生成文件的时间戳
    文件说明 缩写：文件名     | 缩写如下 文件名在各个缩写的文件中
    bzd     knd         qzd     wxg     zc
    半遮挡   没看到车牌    全遮挡  未悬挂   正常
    
    视频每隔50帧截取一张 先截取最新的视频文件 如有不够截取图片张数 将读取之前的视频文件
    车辆检测模型只获取边框都大于250px 并且总像素点数大于62500px 的车辆图片
    50帧 250px以及 62500px都可以根据实际情况进行修改
'''
# frozen_inference_graph.pb中保存的网络的结构跟数据
# own_label_map.pbtxt中保存index到类别名的映射 需要通过pbtxt具体对应的类别是什么
sb_id = {}
IMG_NUMBER = 3  # 一次保存从视频中截取几张图片
NUM_CLASSES = 1  # 这个不要动
PATH_TO_LABELS = 'car_Detection.pbtxt'  # 这个跟下边俩一样 别动
PATH_TO_FROZEN_GRAPH = 'car_Detection.pb'
PATH_TO_Classification = 'License_plate_classification_93.h5'
PATH_TO_VIDEO = '/data/TRAS/video/video/'  # 视频根目录
TEST_IMAGE_PATHS = '/data/TRAS/wusun/'  # 程序根目录


def load_image_into_numpy_array(image_copy):
    im_width, im_height = image_copy.size
    return np.array(image_copy.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# TODO 根据视频截图的图片进行提取车辆以及车牌分类
def main():
    img_list = {}
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            # for path, dirs, files in os.walk(TEST_IMAGE_PATHS):
            for files in sb_id:
                for image_path in sb_id[files]:
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '打开 {} 图片'.format(image_path))
                    image_1 = Image.open(os.path.join(TEST_IMAGE_PATHS, image_path))
                    # 图片转换为numpy的形式
                    print(time.time())
                    image_np = load_image_into_numpy_array(image_1)
                    # 图片扩展一维，最后进入神经网络的图片格式应该为[1, ?, ?, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    print(time.time())
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
                    print(time.time())
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    
                    image2 = Image.fromarray(image_np)
                    # image2.save('{}/{}'.format(output_dict['detection_classes'][0], image_path))
                    for i in range(output_dict['num_detections']):
                        if output_dict['detection_scores'][i] < 0.8:
                            break
                        x_1 = int(image2.size[0] * output_dict['detection_boxes'][i][1])
                        y_1 = int(image2.size[1] * output_dict['detection_boxes'][i][0])
                        x_2 = int(image2.size[0] * output_dict['detection_boxes'][i][3])
                        y_2 = int(image2.size[1] * output_dict['detection_boxes'][i][2])
                        if x_2-x_1 > 250 and y_2-y_1 > 250 and (x_2-x_1)*(y_2-y_1) > 62500:
                            img = image2.crop((x_1, y_1, x_2, y_2)).resize((400, 400), Image.ANTIALIAS)
                            if image_path not in img_list:
                                img_list[image_path] = []
                            img_list[image_path].append(img)
                    shutil.move(os.path.join(TEST_IMAGE_PATHS, image_path), os.path.join(path_list[5], image_path))
    if not img_list:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '当前没有捕获到满足条件的车辆')
        return
    with open(os.path.join(TEST_IMAGE_PATHS, '{}.txt'.format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))), 'w') as f:
        for img_name in img_list:
            value = 0
            for img in img_list[img_name]:
                img_tensor = image.img_to_array(img)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                img_tensor /= 255
                i = model.predict(img_tensor)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '{}_{}_{}.jpg'.format(os.path.splitext(img_name)[0], leibie_list[int(np.where(i == np.max(i))[1])][:-1], value))
                img.save(os.path.join(path_list[int(np.where(i == np.max(i))[1])],
                                      '{}_{}_{}.jpg'.format(os.path.splitext(img_name)[0], leibie_list[int(np.where(i == np.max(i))[1])][:-1], value)))
                f.write('{a} : {}_{a}_{}.jpg\n'.format(os.path.splitext(img_name)[0], value, a=leibie_list[int(np.where(i == np.max(i))[1])][:-1]))
                value += 1


# TODO 从视频中截取照片
def video_to_img():
    global sb_id
    global IMG_NUMBER
    sb_id = {}
    IMG_NUMBER = 3
    for path, dirs, files in os.walk(PATH_TO_VIDEO):
        files = [x for x in files if os.path.splitext(x)[1] == '.dat']  # 筛选出只有该扩展名的视频文件
        for image_path in files:
            if time.time() - os.path.getmtime(os.path.join(path, image_path)) > 60 * 60:
                os.remove(os.path.join(path, image_path))
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
                    cv2.imwrite(os.path.join(TEST_IMAGE_PATHS, os.path.splitext(image_path)[0] + '{}.jpg'.format(IMG_NUMBER)), frame)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                          '生成的图片: ' + os.path.splitext(image_path)[0] + '{}.jpg'.format(IMG_NUMBER))
                    if image_path[:image_path.find('_')] not in sb_id:
                        sb_id[image_path[:image_path.find('_')]] = []
                    sb_id[image_path[:image_path.find('_')]].append(os.path.splitext(image_path)[0] + '{}.jpg'.format(IMG_NUMBER))
                    IMG_NUMBER -= 1
                    if not IMG_NUMBER:
                        cap.release()
                        # os.remove(os.path.join(path, image_path))
                        return
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            # os.remove(os.path.join(path, image_path))


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
    # 打开车牌分类模型
    model = load_model(PATH_TO_Classification)
    leibie_list = ['bzd/', 'knd/', 'qzd/', 'wxg/', 'zc/', 'yt/']
    path_list = []
    for j in range(len(leibie_list)):
        path_list.append(os.path.join(TEST_IMAGE_PATHS, leibie_list[j]))
        if not os.path.exists(path_list[j]):
            os.makedirs(path_list[j])

    timed_task()
