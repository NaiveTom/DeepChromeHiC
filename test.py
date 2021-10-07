# -*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 禁止显示系统信息



# cpu_count = 4



# 清晰的打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免混淆

# 引入全部模型
from model import *

from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score


import tensorflow as tf
import numpy as np
# 打印完整信息
# np.set_printoptions(threshold=np.inf)

from keras.preprocessing.image import ImageDataGenerator
# one-hot encoding
from keras.utils.np_utils import to_categorical



# 因为服务器没有图形界面，所以必须这样弄
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt










def test_cnn_dense_resnet(gen_name, model_name, target_name, gene_length):

    # 打印说明，方便检查
    fancy_print('gen_name', gen_name)
    fancy_print('model_name', model_name)
    fancy_print('target_name', target_name)

    ##############################
    #
    # Test set iterator
    #
    ##############################

    datagen = ImageDataGenerator(rescale = 1./255)

    BATCH_SIZE = 32 # 一次大小

    test_generator = datagen.flow_from_directory(directory = 'data/'+target_name+'/png_test/', target_size = (gene_length*2, 5),
                                                 color_mode = 'grayscale',
                                                 batch_size = BATCH_SIZE,
                                                 shuffle = False) # 不需要 shuffle

    ##############################
    #
    # Load the pre-trained model
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

        
        
    clf.load_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')

    fancy_print('Load the model ...', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')

    ##############################
    #
    # prediction
    #
    ##############################

    # Measure accuracy first
    # score = clf.evaluate_generator(generator = test_generator, steps = len(test_generator))
    # workers = cpu_count, use_multiprocessing = True)
    # 以上不能使用多进程，不然会出错
    # fancy_print('loss & acc', score)

    # Use model.predict to get the predicted probability of the test set
    y_prob = clf.predict_generator(generator = test_generator, steps = len(test_generator))
    # workers = cpu_count, use_multiprocessing = True)
    # 以上不能使用多进程，不然会出错
    # fancy_print('y_prob', y_prob, '.')
    # fancy_print('y_prob.shape', y_prob.shape, '-')

    # Get label
    label_test_tag = test_generator.class_indices
    label_test_name = test_generator.filenames

    # 没有反，这里是对的
    label_test = []
    for i in label_test_name:
        # windows系统
        # label = i.split('\\')[0] # Separate categories
        label = i.split('/')[0] # Separate categories
        # 这里是对的
        label_test.append( int(label_test_tag[label]) ) # Make it into number





    from scipy.stats import pearsonr
    
    pesr = pearsonr( label_test, [ i[1] for i in y_prob ] )

    


    
    label_test = to_categorical(label_test)

    # fancy_print('label_test', label_test, '.')
    # fancy_print('label_test.shape', label_test.shape, '-')
    

    # Calculate ROC curve and AUC for each category
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Two classification problem
    n_classes = label_test.shape[1] # n_classes = 2
    fancy_print('n_classes', n_classes) # n_classes = 2

    # Draw ROC curve using actual category and predicted probability
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fancy_print('fpr', fpr)
    # fancy_print('tpr', tpr)
    fancy_print('roc_auc', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color = 'darkorange', 
             lw = lw, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc = "lower right")

    plt.savefig('result/'+gen_name+'/'+model_name+'/'+target_name+'_rocauc.png')
    # plt.pause(0.01)
    # plt.show()

    fw = open('log.txt','a+')
    import time
    fw.write( time.asctime(time.localtime(time.time())) + '\n' )
    fw.write( gen_name + '\t' + target_name + '\t' + model_name + '\t' + str(roc_auc[1]) + '\t' + str(pesr[0]) + '\n' )
    fw.close()

    fancy_print('roc_auc[1]', roc_auc[1], '-')




















def test_cnn_separate(gen_name, model_name, target_name, gene_length):
    
    ##############################
    #
    # 测试集迭代器
    #
    ##############################

    from keras.preprocessing.image import ImageDataGenerator

    BATCH_SIZE = 32 # 每次大小

    test_datagen = ImageDataGenerator(rescale = 1. / 255)

    # 这个是用来产生label的
    test_generator = test_datagen.flow_from_directory(directory = 'data/'+target_name+'/test_en/', target_size=(gene_length, 5),
                                                      color_mode = 'grayscale',
                                                      class_mode = 'categorical',
                                                      # "categorical"会返回2D的one-hot编码标签, "binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                      batch_size = BATCH_SIZE,
                                                      shuffle = False)  # 不要乱

    def generator_two_test():
        test_generator1 = test_datagen.flow_from_directory(directory = 'data/'+target_name+'/test_en/', target_size = (gene_length, 5),
                                                          color_mode = 'grayscale',
                                                          class_mode = 'categorical', # "categorical"会返回2D的one-hot编码标签, "binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                          batch_size = BATCH_SIZE,
                                                          shuffle = False) # 不要乱

        test_generator2 = test_datagen.flow_from_directory(directory = 'data/'+target_name+'/test_pr/', target_size = (gene_length, 5),
                                                          color_mode = 'grayscale',
                                                          class_mode = 'categorical', # "categorical"会返回2D的one-hot编码标签, "binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                          batch_size = BATCH_SIZE,
                                                          shuffle = False) # 不要乱
        while True:
            out1 = test_generator1.next()
            out2 = test_generator2.next()
            yield [out1[0], out2[0]] # , out1[1]  # 返回两个的组合和结果



    ##############################
    #
    # Loading 已经训练好的模型
    #
    ##############################

    ##############################
    #
    # Model building
    #
    ##############################

    if model_name == 'onehot_cnn_two_branch':
        clf = model_onehot_cnn_two_branch(gene_length)



    # 跳过训练，直接Load the model ...
    from keras.models import load_model
    clf.load_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')

    fancy_print('Load the model ...', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')

    



    ##############################
    #
    # 预测
    #
    ##############################

    # 新加入内容，用来评估模型质量
    # 计算auc和绘制roc_curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 先测算准确度
    # score = clf.evaluate_generator(generator = generator_two_test(), steps = len(test_generator))
    # fancy_print('loss & acc', score)

    # 打印所有内容
    # np.set_printoptions(threshold = np.inf)

    # 利用model.predict获取测试集的预测概率
    y_prob = clf.predict_generator(generator = generator_two_test(), steps = len(test_generator))
    # workers = cpu_count, use_multiprocessing = True)
    # 以上不能使用多进程，不然会出错
    # fancy_print('y_prob', y_prob, '.')
    # fancy_print('y_prob.shape', y_prob.shape, '-')



    # 获得label
    label_test_tag = test_generator.class_indices
    label_test_name = test_generator.filenames

    label_test = []
    for i in label_test_name:
        # windows系统
        # label = i.split('\\')[0] # Separate categories
        label = i.split('/')[0] # Separate categories
        label_test.append(int(label_test_tag[label])) # 弄成数字呗





    from scipy.stats import pearsonr
    
    # print(label_test)
    # print([ i[1] for i in y_prob ])
    
    # print(len(label_test))
    # print(len([ i[1] for i in y_prob ]))
    
    pesr = pearsonr( label_test, [ i[1] for i in y_prob ] )





    from keras.utils.np_utils import to_categorical
    label_test = to_categorical(label_test)

    # fancy_print('label_test', label_test, '.')
    # fancy_print('label_test.shape', label_test.shape, '-')
    



    # 为每个类别计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()



    # 二分类问题
    n_classes = label_test.shape[1] # n_classes = 2
    # fancy_print('n_classes', n_classes) # n_classes = 2

    # 使用实际类别和预测概率绘制ROC曲线
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fancy_print('fpr', fpr)
    # fancy_print('tpr', tpr)
    # fancy_print('cnn_roc_auc', roc_auc)



    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color = 'darkorange', 
             lw = lw, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc = "lower right")

    plt.savefig('result/'+gen_name+'/'+model_name+'/'+target_name+'_rocauc.png')
    # plt.pause(0.01)

    # plt.show()

    fw = open('log.txt','a+')
    import time
    fw.write( time.asctime(time.localtime(time.time())) + '\n' )
    fw.write( gen_name + '\t' + target_name + '\t' + model_name + '\t' + str(roc_auc[1]) + '\t' + str(pesr[0]) + '\n' )
    fw.close()

    fancy_print('roc_auc[1]', roc_auc[1], '-')






















def test_embedding(gen_name, model_name, target_name):

    # 打印说明，方便检查
    fancy_print('gen_name', gen_name)
    fancy_print('model_name', model_name)
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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



    model.load_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')

    fancy_print('Load the model ...', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')

    # Data_dir = '/home/ycm/data/%s/' % name
    test = np.load('data/'+target_name+'/embedding_test.npz')
    fancy_print('Loading npz', 'data/'+target_name+'/embedding_test.npz', '=')
    X_en_tes, X_pr_tes, y_test = test['X_en_tes'], test['X_pr_tes'], test['y_tes']
    # fancy_print('y_test', y_test[:100], '=')

    ##############################
    #
    # prediction
    #
    ##############################

    # 利用model.predict获取测试集的预测值
    y_score = model.predict([X_en_tes, X_pr_tes])
    # workers = cpu_count, use_multiprocessing = True)
    # 以上不能使用多进程，不然会出错
    # fancy_print('y_score', y_score, '.')

    # 计算F1
    # y_pred = y_score.argmax(axis=-1)
    # fancy_print('f1', f1_score(y_test, y_pred, average='binary'), '.')

    # 为每个类别计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()



    # 二进制化输出
    # 转成独热码
    y_test = y_test.tolist()
    y_temp_1 = []; y_temp_0 = []
    for each in y_test:
        if each == 1: y_temp_1.append(1); y_temp_0.append(0)
        else: y_temp_1.append(0); y_temp_0.append(1)
    y_test = [y_temp_0, y_temp_1]; y_test = np.transpose(y_test)
    # fancy_print('y_test', y_test, '.')
    # fancy_print('y_test.shape', y_test.shape)

    # y_test = label_binarize(y_test, classes=[0, 1, 2])  # 二分类
    # fancy_print('y_test', y_test)

    # 二进制化输出
    # 转成独热码，不需要了，这次直接就是两组
    
    y_score = y_score.tolist()
    y_temp_1 = []; y_temp_0 = []
    for each in y_score:
        y_temp_1.append(float(each[0]))
        y_temp_0.append(1-float(each[0]))
    y_score = [y_temp_0, y_temp_1]; y_score = np.transpose(y_score)
    # fancy_print('y_score', y_score)
    # fancy_print('y_score.shape', y_score.shape)
    
    # y_score = label_binarize(y_score, classes=[0, 1, 2])  # 二分类
    # fancy_print('y_score', y_score)





    from scipy.stats import pearsonr
    
    pesr = pearsonr( [ i[1] for i in y_test ], [ i[1] for i in y_score ] )





    n_classes = y_test.shape[1] # n_classes = 2
    # fancy_print('n_classes', n_classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fancy_print('fpr', fpr)
    # fancy_print('tpr', tpr)
    # fancy_print('roc_auc', roc_auc)

    

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.6f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(ROC)')
    plt.legend(loc="lower right")

    plt.savefig('result/'+gen_name+'/'+model_name+'/'+target_name+'_rocauc.png')
    # plt.pause(0.01)

    fw = open('log.txt','a+')
    import time
    fw.write( time.asctime(time.localtime(time.time())) + '\n' )
    fw.write( gen_name + '\t' + target_name + '\t' + model_name + '\t' + str(roc_auc[1]) + '\t' + str(pesr[0]) + '\n' )
    fw.close()

    fancy_print('roc_auc[1]', roc_auc[1], '-')



########################################
#
#   本模块没有代码运行
#
########################################
if __name__ == '__main__':
    
    pass
