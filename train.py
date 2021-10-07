# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Remove unnecessary information



import numpy as np

# cpu_count = 4



# 因为服务器没有图形界面，所以必须这样弄
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



# 好看的打印格式
def fancy_print(n = None, c = None, s = '#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 空一行避免混淆

# 拿到所有模型
from model import *

# 图片读取生成器
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

most_epoches = 500 # 最大训练次数 500 测试时 2-10










def train_cnn_dense_resnet(gen_name, model_name, gene_length):

    # 打印说明，方便检查
    fancy_print('gen_name', gen_name)
    fancy_print('model_name', model_name)

    ##############################
    #
    # png reader in iterator
    #
    ##############################

    # 训练集：验证集：测试集 = 8：1：1
    train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.11) # set validation split

    BATCH_SIZE = 32 # 一次大小

    train_generator = train_datagen.flow_from_directory(directory = 'data/'+gen_name+'/png_train/',
                                                        target_size = (gene_length*2, 5),
                                                        color_mode = 'grayscale',
                                                        class_mode = 'categorical',
                                                        batch_size = BATCH_SIZE,
                                                        subset = 'training',  # set as training data
                                                        shuffle = True, # must shuffle
                                                        seed = 42,
                                                        )
    val_generator = train_datagen.flow_from_directory(directory = 'data/'+gen_name+'/png_train/', # same directory as training data
                                                      target_size = (gene_length*2, 5),
                                                      color_mode = 'grayscale',
                                                      class_mode = 'categorical',
                                                      batch_size = BATCH_SIZE,
                                                      subset = 'validation', # set as validation data
                                                      shuffle = True, # must shuffle
                                                      seed = 42,
                                                      )

    ##############################
    #
    # loss数据可视化
    #
    ##############################

    class PlotProgress(keras.callbacks.Callback):

        def __init__(self, entity = ['loss', 'accuracy']):
            self.entity = entity

        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []

            self.accs = []
            self.val_accs = []

            self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            # 损失函数
            self.losses.append(logs.get('{}'.format(self.entity[0])))
            self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
            # 准确率
            self.accs.append(logs.get('{}'.format(self.entity[1])))
            self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))

            self.i += 1

            # clear_output(wait=True)
            plt.figure(0)
            plt.clf() # 清理历史遗迹
            plt.plot(self.x, self.losses, label="{}".format(self.entity[0]))
            plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
            plt.legend()
            plt.savefig('result/'+gen_name+'/'+model_name+'/loss.png')
            # plt.pause(0.01)
            # plt.show()

            plt.figure(1)
            plt.clf()  # 清理历史遗迹
            plt.plot(self.x, self.accs, label="{}".format(self.entity[1]))
            plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
            plt.legend()
            plt.savefig('result/'+gen_name+'/'+model_name+'/acc.png')
            # plt.pause(0.01)
            # plt.show()

    ##############################
    #
    # Model building
    #
    ##############################

    if model_name == 'onehot_cnn_one_branch':
        clf = model_onehot_cnn_one_branch(gene_length)
    if model_name == 'onehot_embedding_dense':
        clf = model_onehot_embedding_dense(gene_length)

    if model_name == 'onehot_dense':
        clf = model_onehot_dense(gene_length)
    if model_name == 'onehot_resnet18':
        clf = model_onehot_resnet18(gene_length)
    if model_name == 'onehot_resnet34':
        clf = model_onehot_resnet34(gene_length)

    

    clf.summary() # Print model structure

    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)

    # 绘图函数
    plot_progress = PlotProgress(entity = ['loss', 'accuracy'])

    

    ##############################
    #
    # Model training
    #
    ##############################

    # No need to count how many epochs, keras can count
    history = clf.fit_generator(generator = train_generator,
                                epochs = most_epoches,
                                validation_data = val_generator,

                                steps_per_epoch = train_generator.samples // BATCH_SIZE,
                                validation_steps = val_generator.samples // BATCH_SIZE,

                                callbacks = [plot_progress, early_stopping],

                                # max_queue_size = 64,
                                # workers = cpu_count,
                                # use_multiprocessing = True,

                                verbose = 2 # 一次训练就显示一行
                                )

    clf.save_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')
    # 打印一下，方便检查
    fancy_print('save_weights', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')




















def train_cnn_separate(gen_name, model_name, gene_length):
    
    ##############################
    #
    # 构建迭代器
    #
    ##############################

    from keras.preprocessing.image import ImageDataGenerator

    # train_datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rescale = 1. / 255) # 上下翻转 左右翻转
    train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = 0.11)

    BATCH_SIZE = 32 # 每次大小

    def generator_two_train():
        train_generator1 = train_datagen.flow_from_directory(directory = 'data/'+gen_name+'/train_en/', target_size = (gene_length, 5),
                                                             color_mode = 'grayscale',
                                                             class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                             batch_size = BATCH_SIZE,
                                                             subset = 'training',  # set as training data
                                                             shuffle = True,
                                                             seed = 42)  # 相同方式打散
        train_generator2 = train_datagen.flow_from_directory(directory = 'data/'+gen_name+'/train_pr/', target_size = (gene_length, 5),
                                                             color_mode = 'grayscale',
                                                             class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                             batch_size = BATCH_SIZE,
                                                             subset = 'training',  # set as training data
                                                             shuffle = True,
                                                             seed = 42)  # 相同方式打散
        while True:
            out1 = train_generator1.next()
            out2 = train_generator2.next()
            yield [out1[0], out2[0]], out1[1]  # 返回两个的组合和结果

    def generator_two_val():
        val_generator1 = train_datagen.flow_from_directory(directory = 'data/'+gen_name+'/train_en/', target_size = (gene_length, 5),
                                                           color_mode = 'grayscale',
                                                           class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                           batch_size = BATCH_SIZE,
                                                           subset = 'validation', # set as validation data
                                                           shuffle =True,
                                                           seed = 42)  # 相同方式打散
        val_generator2 = train_datagen.flow_from_directory(directory = 'data/'+gen_name+'/train_pr/', target_size = (gene_length, 5),
                                                           color_mode = 'grayscale',
                                                           class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                           batch_size = BATCH_SIZE,
                                                           subset = 'validation', # set as validation data
                                                           shuffle = True,
                                                           seed = 42)  # 相同方式打散
        while True:
            out1 = val_generator1.next()
            out2 = val_generator2.next()
            yield [out1[0], out2[0]], out1[1] # 返回两个的组合和结果

    ##############################
    #
    # 模型搭建
    #
    ##############################

    # 如果出现版本不兼容，那么就用这两句代码，否则会报警告
    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()

    from sklearn import metrics
    from keras.callbacks import ModelCheckpoint



    ##############################
    #
    # Model building
    #
    ##############################

    if model_name == 'onehot_cnn_two_branch':
        clf = model_onehot_cnn_two_branch(gene_length)



    clf.summary() # 打印模型结构

    '''
    filename = 'best_model.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
    '''
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)





    '''
    fancy_print('train_generator.next()[0]', train_generator.next()[0], '+')
    fancy_print('train_generator.next()[1]', train_generator.next()[1], '+')
    fancy_print('train_generator.next()[0].shape', train_generator.next()[0].shape, '+')
    fancy_print('train_generator.next()[1].shape', train_generator.next()[1].shape, '+')

    fancy_print('val_generator.next()[0]', val_generator.next()[0], '-')
    fancy_print('val_generator.next()[1]', val_generator.next()[1], '-')
    fancy_print('val_generator.next()[0].shape', val_generator.next()[0].shape, '-')
    fancy_print('val_generator.next()[1].shape', val_generator.next()[1].shape, '-')
    '''
    ##############################
    #
    # 模型训练
    #
    ##############################

    # 不需要再算多少个epoch了，自己会算
    history = clf.fit_generator(generator = generator_two_train(),
                                epochs = most_epoches,
                                validation_data = generator_two_val(),

                                steps_per_epoch = 24568 * 2 // BATCH_SIZE, # 全部训练
                                validation_steps = 3071 * 2 // BATCH_SIZE, # 全部验证

                                callbacks = [early_stopping],
                                
                                shuffle = True, # 再次 shuffle

                                # max_queue_size = 64,
                                # workers = cpu_count,
                                # use_multiprocessing = True,

                                verbose = 2) # 一次训练就显示一行

    clf.save_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')
    # 打印一下，方便检查
    fancy_print('save_weights', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')




















def train_embedding(gen_name, model_name):

    # 打印说明，方便检查
    fancy_print('gen_name', gen_name)
    fancy_print('model_name', model_name)

    '''
    2021-04-11 16:53:06.007063: E tensorflow/stream_executor/dnn.cc:616] CUDNN_STATUS_INTERNAL_ERROR

    in tensorflow/stream_executor/cuda/cuda_dnn.cc(2011): 'cudnnRNNBackwardData( cudnn.handle(), rnn_desc.handle(), 
    model_dims.max_seq_length, output_desc.handles(), output_data.opaque(), output_desc.handles(), output_backprop_data.opaque(), 
    output_h_desc.handle(), output_h_backprop_data.opaque(), output_c_desc.handle(), output_c_backprop_data.opaque(), 
    rnn_desc.params_handle(), params.opaque(), input_h_desc.handle(), input_h_data.opaque(), input_c_desc.handle(), 
    input_c_data.opaque(), input_desc.handles(), input_backprop_data->opaque(), input_h_desc.handle(), input_h_backprop_data->opaque(), 
    input_c_desc.handle(), input_c_backprop_data->opaque(), workspace.opaque(), workspace.size(), reserve_space_data->opaque(), reserve_space_data->size())'

    2021-04-11 16:53:06.007530: W tensorflow/core/framework/op_kernel.cc:1767] OP_REQUIRES failed at cudnn_rnn_ops.cc:1922: 
    Internal: Failed to call ThenRnnBackward with model config: [rnn_mode, rnn_input_mode, rnn_direction_mode]: 3, 0, 0 , 
    [num_layers, input_size, num_units, dir_count, max_seq_length, batch_size, cell_num_units]: [1, 64, 50, 1, 100, 32, 0] 

    2021-04-11 16:53:06.007077: F tensorflow/stream_executor/cuda/cuda_dnn.cc:190] Check failed: status == CUDNN_STATUS_SUCCESS (7 vs. 0)Failed to set cuDNN stream.

    解决方案
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ##############################
    #
    # loss数据可视化
    #
    ##############################

    class PlotProgress(keras.callbacks.Callback):

        def __init__(self, entity = ['loss', 'accuracy']):
            self.entity = entity

        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []

            self.accs = []
            self.val_accs = []

            self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            # 损失函数
            self.losses.append(logs.get('{}'.format(self.entity[0])))
            self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
            # 准确率
            self.accs.append(logs.get('{}'.format(self.entity[1])))
            self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))

            self.i += 1


            plt.figure(0)
            plt.clf() # 清理历史遗迹
            plt.plot(self.x, self.losses, label="{}".format(self.entity[0]))
            plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
            plt.legend()
            plt.savefig('result/'+gen_name+'/'+model_name+'/loss.png')
            # plt.pause(0.01)
            # plt.show()

            plt.figure(1)
            plt.clf()  # 清理历史遗迹
            plt.plot(self.x, self.accs, label="{}".format(self.entity[1]))
            plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
            plt.legend()
            plt.savefig('result/'+gen_name+'/'+model_name+'/acc.png')
            # plt.pause(0.01)
            # plt.show()



    train = np.load('data/'+gen_name+'/embedding_train.npz')
    X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']

    ##############################
    #
    # Model building
    #
    ##############################
    
    if model_name == 'embedding_cnn_one_branch':
        model = model_embedding_cnn_one_branch()
    if model_name == 'embedding_cnn_two_branch':
        model = model_embedding_cnn_two_branch()
    if model_name == 'embedding_dense':
        model = model_embedding_dense()

    if model_name == 'onehot_embedding_cnn_one_branch':
        model = model_onehot_embedding_cnn_one_branch()
    if model_name == 'onehot_embedding_cnn_two_branch':
        model = model_onehot_embedding_cnn_two_branch()


        
    model.summary()

    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 20, restore_best_weights = True)
    # 绘图函数
    plot_progress = PlotProgress(entity = ['loss', 'accuracy'])

    

    history = model.fit([X_en_tra, X_pr_tra], y_tra, epochs=most_epoches, batch_size=32, validation_split=0.11,
                        callbacks=[early_stopping, plot_progress],

                        # max_queue_size = 64,
                        # workers = cpu_count,
                        # use_multiprocessing = True,
                        
                        verbose = 2 # 一次训练就显示一行
                        )

    model.save_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')
    # 打印一下，方便检查
    fancy_print('save_weights', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')



########################################
#
#   本模块没有代码运行
#
########################################
if __name__ == '__main__':
    
    pass
