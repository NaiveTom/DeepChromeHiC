# -*- coding:utf-8 -*-

import keras
from keras import Sequential, initializers
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

import numpy as np

from lib.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152



MAX_LEN = 10001










##############################
#
# onehot
#
##############################










def model_onehot_cnn_one_branch(gene_length):

    model = Sequential()
     
    model.add(Conv2D(64, kernel_size=(24, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', activation = 'relu', input_shape=((gene_length*2, 5, 1))))
    model.add(BatchNormalization())
        
    model.add(Conv2D(64, kernel_size=(24, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(24, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(24, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', activation = 'relu'))
    model.add(BatchNormalization())
        
    model.add(Conv2D(128, kernel_size=(24, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(24, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
      
    model.add(Flatten())
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform')) # 最后不加 Dropout
    model.add(Dense(2, activation='softmax')) # 输出层：softmax 归一化

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 5e-5, decay = 1e-5),
                  metrics = ['accuracy'])

    return model




















def model_onehot_cnn_two_branch(gene_length):
        
    # 模型1
    input_1 = Input(shape = (gene_length, 5, 1))
  
    model_1 = Conv2D(64, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(input_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(64, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(64, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = BatchNormalization()(model_1)
  
    model_1 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_1)
    model_1 = BatchNormalization()(model_1)
  
    model_1 = Conv2D(128, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(128, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Conv2D(128, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = BatchNormalization()(model_1)
  
    model_1 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_1)
  
    model_1 = Flatten()(model_1)
    model_1 = BatchNormalization()(model_1)
  
    model_1 = Dense(512, activation = 'relu', kernel_initializer='glorot_uniform')(model_1)
    model_1 = Dropout(0.5)(model_1)



    # 模型2
    input_2 = Input(shape = (gene_length, 5, 1))
  
    model_2 = Conv2D(64, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(input_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Conv2D(64, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Conv2D(64, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = BatchNormalization()(model_2)
  
    model_2 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_2)
    model_2 = BatchNormalization()(model_2)
  
    model_2 = Conv2D(128, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Conv2D(128, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Conv2D(128, kernel_size = (24, 1), strides = (4, 1), kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = BatchNormalization()(model_2)
  
    model_2 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_2)
  
    model_2 = Flatten()(model_2)
    model_2 = BatchNormalization()(model_2)
  
    model_2 = Dense(512, activation = 'relu', kernel_initializer='glorot_uniform')(model_2)
    model_2 = Dropout(0.5)(model_2)



    # 合并模型
    merge = concatenate([model_1, model_2] , axis=-1)
    
    merge = Dense(512, activation = 'relu', kernel_initializer='glorot_uniform')(merge)
    merge = Dropout(0.5)(merge)
    
    merge = Dense(512, activation = 'relu', kernel_initializer='glorot_uniform')(merge)
    merge = Dense(2, activation = 'softmax')(merge)

    model = Model(inputs = [input_1, input_2], outputs = merge)

    model.compile(loss = 'categorical_crossentropy', # sparse_categorical_crossentropy binary_crossentropy
                  optimizer = Adam(lr = 5e-5, decay = 1e-5),
                  metrics = ['accuracy'])
  
    return model




















def model_onehot_embedding_dense(gene_length):
      
    model = Sequential()

    # embedding层
    model.add(Embedding(4097, 8, input_length=20002, trainable=True, input_shape=((gene_length*2, 5, 1))))

   

    model.add(Flatten())
    model.add(BatchNormalization())

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(2, activation='softmax')) # 输出层：softmax 归一化

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 5e-5, decay = 1e-5),
                  metrics = ['accuracy'])

    return model





def model_onehot_dense(gene_length):
  
    model = Sequential()

    model.add( Flatten( input_shape=((gene_length*2, 5, 1)) ) )
    model.add(BatchNormalization())
  
    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(2, activation='softmax')) # 输出层：softmax 归一化

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 5e-5, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model





def model_onehot_resnet18(gene_length):

    model = resnet_18() # ResNet152网络，这样的话会训练的无比慢，所以建议用50或者101

    # 需要至少32GB显存，根本没法调试
    """   
    model = resnet_50()
    model = resnet_101()
    model = resnet_152()
    """

    model.build( input_shape = (None, gene_length*2, 5, 1) )

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 5e-5, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model





def model_onehot_resnet34(gene_length):

    model = resnet_34() # ResNet152网络，这样的话会训练的无比慢，所以建议用50或者101

    # 需要至少32GB显存，根本没法调试
    """
    model = resnet_50()
    model = resnet_101()
    model = resnet_152()
    """

    model.build( input_shape = (None, gene_length*2, 5, 1) )

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 5e-5, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model










##############################
#
# embedding
#
##############################










def model_embedding_cnn_one_branch():

    embedding_matrix = np.load('embedding_matrix.npy')

    ####################
    # 输入部分
    ####################
    region_1 = Input(shape=(MAX_LEN,))
    region_2 = Input(shape=(MAX_LEN,))

    ####################
    # embedding 部分
    ####################
    embedding_region_1 = Embedding(4097, 100, weights=[embedding_matrix], trainable=True)(region_1)
    embedding_region_2 = Embedding(4097, 100, weights=[embedding_matrix], trainable=True)(region_2)

    ####################
    # 合并部分
    ####################
    merge_layer = Concatenate(axis = 1)([embedding_region_1, embedding_region_2])

    ####################
    # cnn部分
    ####################
    conv_layer = Convolution1D(filters=64,
                               kernel_size=32,
                               strides = 8,
                               padding='same',
                               kernel_initializer='he_normal',
                               )
    max_pool_layer = MaxPooling1D(pool_size=32, strides=16)



    clf = Sequential()
    
    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)

    clf.add(BatchNormalization())

    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)
    
    clf.add(BatchNormalization())

    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)

    clf.add(BatchNormalization())

    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)
    
    clf.add(BatchNormalization())
    
    clf_out = clf(merge_layer)
    clf_out = Flatten()(clf_out)



    hidden = BatchNormalization()(clf_out)
    hidden = Dropout(0.5)(hidden)
    
    ####################
    # dense 部分
    ####################
    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)

    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    preds = Dense(1, activation='sigmoid')(hidden)
    model = Model([region_1, region_2], preds)
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 5e-6, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model





def model_embedding_cnn_two_branch():

    embedding_matrix = np.load('embedding_matrix.npy')

    ####################
    # 输入部分
    ####################
    region_1 = Input(shape=(MAX_LEN,))
    region_2 = Input(shape=(MAX_LEN,))

    ####################
    # embedding 部分
    ####################
    embedding_region_1 = Embedding(4097, 100, weights=[embedding_matrix], trainable=True)(region_1)
    embedding_region_2 = Embedding(4097, 100, weights=[embedding_matrix], trainable=True)(region_2)

    ####################
    # enhancer 输入部分
    ####################
    region_1_cnn = Convolution1D(filters=64,
                                 kernel_size=32,
                                 strides = 8,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 )
    region_1_pool = MaxPooling1D(pool_size=32, strides=16)

    # enhancer 部分
    region_1_branch = Sequential()
    
    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)

    region_1_branch.add(BatchNormalization())
    
    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)
    
    region_1_branch.add(BatchNormalization())

    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)

    region_1_branch.add(BatchNormalization())
    
    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)
    
    region_1_branch.add(BatchNormalization())
    
    # region_1_branch.add(Dropout(0.5))
    region_1_out = region_1_branch(embedding_region_1)
    region_1_out = Flatten()(region_1_out)

    ####################
    # promoter 输入部分
    ####################
    region_2_cnn = Convolution1D(filters=64,
                                 kernel_size=32,
                                 strides = 8,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 )
    region_2_pool = MaxPooling1D(pool_size=32, strides=16)

    # promoter 部分
    region_2_branch = Sequential()
    
    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)

    region_2_branch.add(BatchNormalization())

    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)
    
    region_2_branch.add(BatchNormalization())

    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)

    region_2_branch.add(BatchNormalization())

    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)
    
    region_2_branch.add(BatchNormalization())
    
    # region_2_branch.add(Dropout(0.5))
    region_2_out = region_2_branch(embedding_region_2)
    region_2_out = Flatten()(region_2_out)

    ####################
    # 合并部分
    ####################
    merge_layer = Concatenate()([region_1_out, region_2_out])

    hidden = BatchNormalization()(merge_layer)
    hidden = Dropout(0.5)(hidden)
    
    ####################
    # dense 部分
    ####################
    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)

    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    preds = Dense(1, activation='sigmoid')(hidden)
    model = Model([region_1, region_2], preds)
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 5e-6, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model





def model_embedding_dense():

    embedding_matrix = np.load('embedding_matrix.npy')

    ####################
    # 输入部分
    ####################
    region_1 = Input(shape=(MAX_LEN,))
    region_2 = Input(shape=(MAX_LEN,))

    ####################
    # embedding 部分
    ####################
    embedding_region_1 = Embedding(4097, 100, weights=[embedding_matrix], trainable=True)(region_1)
    embedding_region_2 = Embedding(4097, 100, weights=[embedding_matrix], trainable=True)(region_2)

    ####################
    # 合并部分
    ####################
    merge_layer = Concatenate(axis = 1)([embedding_region_1, embedding_region_2])

    # dense层避免内存崩溃
    merge_layer = Convolution1D(64, kernel_size=64, strides=32, kernel_initializer='he_uniform', padding='same', activation='relu')(merge_layer)
    merge_layer = MaxPooling1D(pool_size=32, strides=16)(merge_layer)
    merge_layer = BatchNormalization()(merge_layer)
    
    merge_layer = Flatten()(merge_layer)
    merge_layer = Dropout(0.5)((merge_layer))
    
    ####################
    # dense 部分
    ####################
    hidden = Dense(512, kernel_initializer='glorot_uniform')(merge_layer)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)

    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)

    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    preds = Dense(1, activation='sigmoid')(hidden)
    model = Model([region_1, region_2], preds)
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 5e-6, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model





##############################
#
# onthot embedding
#
##############################





def model_onehot_embedding_cnn_one_branch():

    ####################
    # ACGT
    ####################
    embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])

    ####################
    # 输入部分
    ####################
    region_1 = Input(shape=(MAX_LEN,))
    region_2 = Input(shape=(MAX_LEN,))

    ####################
    # embedding 部分
    ####################
    embedding_region_1 = Embedding(4097, 100, input_length=20002, trainable=True)(region_1)
    embedding_region_2 = Embedding(4097, 100, input_length=20002, trainable=True)(region_2)

    ####################
    # 合并部分
    ####################
    merge_layer = Concatenate(axis = 1)([embedding_region_1, embedding_region_2])

    ####################
    # cnn部分
    ####################
    conv_layer = Convolution1D(filters=64,
                               kernel_size=32,
                               strides = 8,
                               padding='same',
                               kernel_initializer='he_normal',
                               )
    max_pool_layer = MaxPooling1D(pool_size=32, strides=16)



    clf = Sequential()
    
    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)

    clf.add(BatchNormalization())

    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)
    
    clf.add(BatchNormalization())

    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)

    clf.add(BatchNormalization())

    clf.add(conv_layer)
    clf.add(Activation('relu'))
    clf.add(max_pool_layer)
    
    clf.add(BatchNormalization())
    
    clf_out = clf(merge_layer)
    clf_out = Flatten()(clf_out)



    hidden = BatchNormalization()(clf_out)
    hidden = Dropout(0.5)(hidden)
    
    ####################
    # dense 部分
    ####################
    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)

    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    preds = Dense(1, activation='sigmoid')(hidden)
    model = Model([region_1, region_2], preds)
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 5e-6, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model





def model_onehot_embedding_cnn_two_branch():

    ####################
    # ACGT
    ####################
    embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])

    ####################
    # 输入部分
    ####################
    region_1 = Input(shape=(MAX_LEN,))
    region_2 = Input(shape=(MAX_LEN,))

    ####################
    # embedding 部分
    ####################
    embedding_region_1 = Embedding(4097, 100, input_length=20002, trainable=True)(region_1)
    embedding_region_2 = Embedding(4097, 100, input_length=20002, trainable=True)(region_2)

    ####################
    # enhancer 输入部分
    ####################
    region_1_cnn = Convolution1D(filters=64,
                                 kernel_size=32,
                                 strides = 8,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 )
    region_1_pool = MaxPooling1D(pool_size=32, strides=16)

    # enhancer 部分
    region_1_branch = Sequential()
    
    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)

    region_1_branch.add(BatchNormalization())

    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)
    
    region_1_branch.add(BatchNormalization())

    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)

    region_1_branch.add(BatchNormalization())

    region_1_branch.add(region_1_cnn)
    region_1_branch.add(Activation('relu'))
    region_1_branch.add(region_1_pool)
    
    region_1_branch.add(BatchNormalization())
    
    # region_1_branch.add(Dropout(0.5))
    region_1_out = region_1_branch(embedding_region_1)
    region_1_out = Flatten()(region_1_out)

    ####################
    # promoter 输入部分
    ####################
    region_2_cnn = Convolution1D(filters=64,
                                 kernel_size=32,
                                 strides = 8,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 )
    region_2_pool = MaxPooling1D(pool_size=32, strides=16)

    # promoter 部分
    region_2_branch = Sequential()
    
    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)

    region_2_branch.add(BatchNormalization())

    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)
    
    region_2_branch.add(BatchNormalization())

    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)

    region_2_branch.add(BatchNormalization())

    region_2_branch.add(region_2_cnn)
    region_2_branch.add(Activation('relu'))
    region_2_branch.add(region_2_pool)
    
    region_2_branch.add(BatchNormalization())
    
    # region_2_branch.add(Dropout(0.5))
    region_2_out = region_2_branch(embedding_region_2)
    region_2_out = Flatten()(region_2_out)

    ####################
    # 合并部分
    ####################
    merge_layer = Concatenate()([region_1_out, region_2_out])

    hidden = BatchNormalization()(merge_layer)
    hidden = Dropout(0.5)(hidden)
    
    ####################
    # dense 部分
    ####################
    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)

    hidden = Dense(512, kernel_initializer='glorot_uniform')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    preds = Dense(1, activation='sigmoid')(hidden)
    model = Model([region_1, region_2], preds)
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 5e-6, decay = 1e-5),
                  metrics = ['accuracy'])
    
    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':

    # 放入需要检查的函数并进行检查
    test_model = model_onehot_embedding_cnn_two_branch()
    name = model_onehot_embedding_cnn_two_branch.__name__

    from keras.utils import plot_model
    plot_model(test_model, to_file = name + '.png')
