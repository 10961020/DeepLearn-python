# !/usr/bin/python  
# encoding: utf-8
# author: zhangtong

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as k
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
'''
    风格迁移
'''

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return k.sum(k.square(combination - base))


def gram_matrix(x):
    features = k.batch_flatten(k.permute_dimensions(x, (2, 0, 1)))
    gram = k.dot(features, k.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return k.sum(k.square(S - C)/4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = k.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = k.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return k.sum(k.pow(a+b, 1.25))

target_image_path = 'C:/Users/Public/Pictures/Sample Pictures/IMG_2959.JPG'  # 想要变换的图片路径
style_reference_image_path = 'C:/Users/Public/Pictures/Sample Pictures/20181205145534.JPG'  # 风格图像的路径

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width*img_height/height)

target_image = k.constant(preprocess_image(target_image_path))
style_reference_image = k.constant(preprocess_image(style_reference_image_path))
combination_image = k.placeholder((1, img_height, img_width, 3))
input_tensor = k.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = k.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_featuers = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_featuers, combination_features)
    loss += (style_weight/len(style_layers))*sl

loss += total_variation_weight * total_variation_loss(combination_image)

grads = k.gradients(loss, combination_image)[0]
fetch_loss_and_grads = k.function([combination_image], [loss, grads])


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values

evaluator = Evaluator()

result_prefix = 'my_result'
iterations = 20

x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration ', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as ', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time-start_time))




































































