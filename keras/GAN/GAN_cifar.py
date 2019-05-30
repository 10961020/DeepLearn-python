# !/usr/bin/python
# encoding: utf-8
# Author: zhangtong
# Time: 2019/5/28 10:27

'''
    使用cifar10数据集 首次使用需要下载cifar10 运行之后自动下载，下载完成之后再次运行不会重新下载
    运行前需要在同级目录创建名为 your_dir 文件夹
    参考 python深度学习 GAN使用 
    选择生成的图案是青蛙
    生成的效果图见your_dir文件夹内
'''
import os
import keras
from keras import layers
import numpy as np
from keras.preprocessing import image

latent_dim = 32
height = 32
width = 32
channles = 3

iterations = 50000
batch_size = 20
save_dir = 'your_dir'
# 稀疏的梯度会妨碍GAN的训练 导致梯度稀疏的两件事:最大池化和relu
# 卷积核的大小要能被步幅整除 防止出现棋盘状伪影 这是由于生成器中的像素空间的不均匀覆盖导致的
# 生成器
generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(128*16*16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)  # 将输入转换为大小为16*16的128个通道的特征图

x = layers.Conv2D(256, 5, padding='same')(x)  # 生成器使用填充方式为same
x = layers.LeakyReLU()(x)  # 激活函数不用relu 选用leakyrelu

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)  # 转置卷积 计算方式 (输入大小-1)*步长-2*填充大小+卷积核大小+输出填充 == 输出大小
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channles, 7, activation='tanh', padding='same')(x)  # 生成器最后一层激活函数用 tanh
generator = keras.models.Model(generator_input, x)
generator.summary()
# 判别器
discriminator_input = layers.Input((height, width, channles))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
# 随机性能能够提高稳健性 训练GAN得到的是一个动态平衡，所以GAN可能以各种方式‘卡住’，随机性有助于防止这种情况发生
# 例如:判别器中使用dropout 另一种向判别器的标签添加随机噪声
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
# decay使用学习率衰减，为了稳定训练过程;clipvalue在优化器中使用梯度裁剪,限制梯度值的范围
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.001, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False  # 因为训练gan将会更新生成器的权重，使得判别器在观察假图像时更有可能预测为真。所以在训练过程中需要将判别器的权重冻结
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# 至此模型搭建完毕·························································
'''
    训练流程大致如下，每轮都进行一下操作
    1.从潜在空间中抽取随机的点(随机噪声)
    2.利用这个随机噪声用generator生成图像
    3.将生成图片与真实图片混合，用混合图片以及相应的标签训练discriminator
    4.在潜在空间抽取新的点
    5.使用这些随机向量以及全部‘真实图片’的标签来训练gan，这个过程是训练生成器去欺骗判别器
'''
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
# reshape第一个参数相加 纬度叠加 如 (1,)+(32,32,3) == (1,32,32,3)
x_train = x_train.reshape((x_train.shape[0],)+(height, width, channles)).astype('float32')/255.

start = 0
for step in range(iterations):
    # 1.
    random_lantent_vectors = np.random.normal(size=(batch_size, latent_dim))  # 使用正态分布对潜在空间中的点进行采样
    # 2.
    generated_images = generator.predict(random_lantent_vectors)  # 生成虚假图片
    # 3.
    stop = start+batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    labels += 0.05*np.random.random(labels.shape)  # 在标签中添加随机噪声
    d_loss = discriminator.train_on_batch(combined_images, labels)  # 训练判别器
    # 4.
    random_lantent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))  # 合成标签，全部是‘真的’
    # 5.
    a_loss = gan.train_on_batch(random_lantent_vectors, misleading_targets)  # 训练gan

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
 
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        generator.save_weights('generator.h5')
        # 训练时如果gan的损失开始增大，而判别器的损失趋近于0，即判别器最终支配了生成器， 需要尝试减小判别器的学习率并增大判别dropout的比率
        print('discriminator loss: ', d_loss)
        print('adversatial loss: ', a_loss)

        img = image.array_to_img(generated_images[0]*255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog'+str(step)+'.png'))
        img = image.array_to_img(real_images[0]*255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog'+str(step)+'.png'))
