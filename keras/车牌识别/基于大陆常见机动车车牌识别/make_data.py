# !/usr/bin/python  
# encoding: utf-8
# Author: zhangtong
# Time: 2019/5/7 11:37

import os
import cv2
import glob
import numpy as np
'''
     根据本地车牌数据制作预训练文件
     本地车牌数据图片的样式 参考同级存放的图片名称及内容
'''


output_file = '1'
validation_percentage = 20  # 以下虽然是8 1 1区分训练验证集 主要是为了尽量先填满 验证测试集 保证训练集可以被32整除 结果比例也会是8 1 1
test_percentage = 20        # 如果训练每次迭代读取的数据不想使用32个 需要把此代码里32都替换掉 比例 如需调整 修改 .8-->到你想要的比例大小


# 读取数据集并分割为训练数据，验证数据和测试数据
def create_image_lists(testing_percentage, validating_percentage):
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    len_images = len(glob.glob('D:/1/project/plate_shibie/可以使用的/*.jpg'))  # 本地车牌图片的绝对路径
    len_images = len_images - 32 * (int(len_images * .8) // 32)  # 验证测试集总数  修改此处  .8
    
    for file_name in glob.glob('D:/1/project/plate_shibie/可以使用的/*.jpg'):  # 本地车牌图片的绝对路径
        dir_name = os.path.basename(file_name)
        print(os.path.splitext(dir_name)[0][:7])
        img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (132, 42))  # 网络模型输入格式大小 可以修改此处 对应训练代码输入也要调整
        # img_tensor = image.img_to_array(img)
        # img_tensor = np.expand_dims(img_tensor, axis=0)
        if len(validation_labels+testing_labels) == len_images:  # 保证train集总数可以被32整除 
            training_images.append(img)
            training_labels.append(os.path.splitext(dir_name)[0][:7])
            continue

        chance = np.random.randint(100)  # 随机  8 1 1比例分配训练验证测试集
        if chance < validating_percentage:
            validation_images.append(img)
            validation_labels.append(os.path.splitext(dir_name)[0][:7])
        elif chance < (validating_percentage+testing_percentage):
            testing_images.append(img)
            testing_labels.append(os.path.splitext(dir_name)[0][:7])
        else:
            training_images.append(img)
            training_labels.append(os.path.splitext(dir_name)[0][:7])

    state = np.random.get_state()  # 数据集随机打乱以获取更好的训练效果
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    print('training_labels length:\t', len(training_labels))
    print('validation_labels length:\t', len(validation_labels))
    print('testing_labels length:\t', len(testing_labels))
    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    processed_data = create_image_lists(test_percentage, validation_percentage)
    np.save(output_file, processed_data)

main()

# processed_data = np.load('1.npy')
# training_images = processed_data[0]
# print(type(training_images))
# X = np.array(training_images, dtype=np.uint8)
# print(X.shape)
