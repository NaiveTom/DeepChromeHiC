# -*- coding:utf-8 -*-

# 第一个文件：residual_block.py

import tensorflow as tf

KERNEL_SIZE = 24 # 基本块大小24
STRIDES = 2 # 隔开2



# 基本块：18和34使用的块，包含两个3*3卷积层
# 然后弄一下大小匹配，不匹配会报错
class BasicBlock(tf.keras.layers.Layer):

    # 内部结构
    def __init__(self, filter_num, stride = 1):
        
        super(BasicBlock, self).__init__()

        # 第一个24*1卷积层
        self.conv1 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (KERNEL_SIZE, 1),
                                            strides = STRIDES,
                                            kernel_initializer = 'he_uniform',
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        # 第二个24*1卷积层
        self.conv2 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (KERNEL_SIZE, 1),
                                            strides = STRIDES,
                                            kernel_initializer = 'he_uniform',
                                            padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 通过1*1卷积层进行shape匹配
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters = filter_num,
                                                       kernel_size = (1, 1),
                                                       strides = stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())

        # shape匹配，直接短接
        else:
            self.downsample = lambda x: x



    def call(self, inputs, training = None, **kwargs):

        # 通过identity模块
        identity = self.downsample(inputs)

        # [b, h, w, c]，通过第一个卷积单元
        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        
        # 通过第二个卷积单元
        x = self.conv2(x)
        x = self.bn2(x, training = training)

        # 2条路径输出直接相加，再通过激活函数
        # output = tf.nn.relu(tf.keras.layers.add([identity, x]))
        output = tf.nn.relu(x)

        return output



# 瓶颈块：50和101和152使用的块，包含三个卷积层，分别是1*1，3*1，1*1（4x）
# 然后弄一下大小匹配，不匹配会报错
class BottleNeck(tf.keras.layers.Layer):
    
    def __init__(self, filter_num, stride = 1):
        
        super(BottleNeck, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (1, 1),
                                            strides = 1,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (3, 3),
                                            strides = stride,
                                            padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters = filter_num * 4,
                                            kernel_size = (1, 1),
                                            strides = 1,
                                            padding = 'same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters = filter_num * 4,
                                                   kernel_size = (1, 1),
                                                   strides = stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())



    def call(self, inputs, training = None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training = training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output



# 创建一个基本块：18和34使用的块
def make_basic_block_layer(filter_num, blocks, stride = 1):
    
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride = stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride = 1))

    return res_block



# 创建一个瓶颈块：50和101和152使用的块
def make_bottleneck_layer(filter_num, blocks, stride = 1):
    
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride = stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride = 1))

    return res_block
