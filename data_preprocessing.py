# -*- coding:utf-8 -*-

import numpy as np
import os





# 仅测试使用，用户可以指定测试的长度
# Use all: SAMPLE = None
# Use 5000: SAMPLE = 5000
SAMPLE = None

# 用于打印的函数，清晰的格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 空一行避免混淆



##############################
#
# Read genetic data and add spaces for later processing
#
##############################

def read_data(name, file_dir): # name用于打印名称

    # Read data
    f = open(file_dir, 'r')
    data = f.readlines()

    data = data[ : SAMPLE ] # Divide a smaller size for testing, if it is None, then select all

    # Replace with a split format, and remove the line break
    for num in range( len(data) ):
        data[num] = data[num].replace('A', '0 ').replace('C', '1 ').replace('G', '2 ') \
                    .replace('T', '3 ').replace('N', '4 ').replace('\n', '')

    f.close()
        
    fancy_print(name + '.shape', np.array(data).shape, '=')

    return np.array(data)

##############################
#
# Split data set into test set, validation set, training set
#
##############################

def data_split(name, data):

    train_split_rate = 0.9 # 0.8 : 0.1 : 0.1
    
    print('-' * 40); print(name); print()

    print('train_split_rate', train_split_rate)
    print('test_split_rate', 1 - train_split_rate)
    
    print()
    
    import math
    
    length = math.floor( len(data) ) # Get the length
    train = data[ : int(length * train_split_rate) ]
    test = data[ int(length * train_split_rate) : ]

    print('len(train_set)', len(train))
    print('len(test_set)', len(test))
    
    print('-' * 40); print()
    
    return train, test

##############################
#
# onehot enconding
#
##############################

# The first parameter is the data to be encoded, and the second parameter is OneHotEncoder
# ACGTN是类别数
def onehot_func(data, ACGTN):

    from keras.utils import to_categorical

    data_onehot = []
    for i in range(len(data)):
        data_onehot.append( np.transpose(to_categorical(data[i].split(), ACGTN)) )

    data_onehot = np.array(data_onehot)

    return data_onehot

##############################
#
# Call function for information processing
#
##############################

def data_process(gen_name):

    ####################
    # Read genetic data
    ####################

    anchor1_pos_raw = read_data('anchor1_pos', 'data/' + gen_name + '/seq.anchor1.pos.txt')
    anchor1_neg2_raw = read_data('anchor1_neg2', 'data/' + gen_name + '/seq.anchor1.neg2.txt')
    anchor2_pos_raw = read_data('anchor2_pos', 'data/' + gen_name + '/seq.anchor2.pos.txt')
    anchor2_neg2_raw = read_data('anchor2_neg2', 'data/' + gen_name + '/seq.anchor2.neg2.txt')

    # 弄成一样长的
    cut_pos = min( len(anchor1_pos_raw), len(anchor2_pos_raw) )
    cut_neg = min( len(anchor1_neg2_raw), len(anchor2_neg2_raw) )
    cut = min( cut_pos, cut_neg )
    
    anchor1_pos_raw = anchor1_pos_raw[:cut]
    anchor2_pos_raw = anchor2_pos_raw[:cut]

    anchor1_neg2_raw = anchor1_neg2_raw[:cut]
    anchor2_neg2_raw = anchor2_neg2_raw[:cut]

    

    ####################
    # shuffle 数据
    ####################
    
    if SAMPLE == None: # 全部混洗
        index = np.random.choice(anchor1_pos_raw.shape[0], size = anchor1_pos_raw.shape[0], replace = False)
        # fancy_print('index_size = anchor1_pos_raw.shape[0]', anchor1_pos_raw.shape[0])
    else: # 混洗一部分，后面的没了，提高效率
        index = np.random.choice(SAMPLE, size = SAMPLE, replace = False)
        # fancy_print('index_size = SAMPLE', SAMPLE)

    anchor1_pos = anchor1_pos_raw[index]
    anchor2_pos = anchor2_pos_raw[index]
    anchor1_neg2 = anchor1_neg2_raw[index]
    anchor2_neg2 = anchor2_neg2_raw[index]

    
    
    ####################
    # Call function for split processing
    ####################

    anchor1_pos_train, anchor1_pos_test = data_split('anchor1_pos', anchor1_pos)
    anchor1_neg2_train, anchor1_neg2_test = data_split('anchor1_neg2', anchor1_neg2)
    anchor2_pos_train, anchor2_pos_test = data_split('anchor2_pos', anchor2_pos)
    anchor2_neg2_train, anchor2_neg2_test = data_split('anchor2_neg2', anchor2_neg2)

    

    ##############################
    #
    # Write picture
    #
    ##############################

    # Write picture module
    # pip install imageio
    import imageio
    from skimage import img_as_ubyte

    # Convert to onehot encoding
    from keras.utils import to_categorical

    ACGTN = 5 # 类别数量

    fancy_print('one-hot enconding',
                '[ [A], [C], [G], [T], [N] ]\n' + str(to_categorical(['0', '1', '2', '3', '4'], ACGTN)))

    ####################
    # Save the training set as a picture
    ####################

    LEN_PER_LOAD = 1000 # The smaller the faster, 1000 is just right

    pic_num = 0

    # Here it is processed in blocks, processing 1000 at a time, because onehot is a second-order complexity, don’t make it too big
    for i in range( int(len(anchor1_pos_train)/LEN_PER_LOAD)+1 ):

        # Show the percentage
        print()
        import time
        print( '>>> ' + time.asctime(time.localtime(time.time())) + '\n' )
        print('The training set and labels are being stored in blocks, block number =', str(i), '/', int( len(anchor1_pos_train)/LEN_PER_LOAD) )

        if (i+1)*LEN_PER_LOAD > len(anchor1_pos_train): # This code deals with the little tail(ending block) problem

            try: # Maybe the little tail(ending block) is 0
                anchor1_pos_train_onehot = onehot_func( anchor1_pos_train[ i*LEN_PER_LOAD : ], ACGTN )
                anchor1_neg2_train_onehot = onehot_func( anchor1_neg2_train[ i*LEN_PER_LOAD : ], ACGTN )
                anchor2_pos_train_onehot = onehot_func( anchor2_pos_train[ i*LEN_PER_LOAD : ], ACGTN )
                anchor2_neg2_train_onehot = onehot_func( anchor2_neg2_train[ i*LEN_PER_LOAD : ], ACGTN )
            except:
                print('The size of the last block is 0, this block has been skipped (to avoid errors)')

        else: # This code handles the normal blocking split
            
            anchor1_pos_train_onehot = onehot_func( anchor1_pos_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )
            anchor1_neg2_train_onehot = onehot_func( anchor1_neg2_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )
            anchor2_pos_train_onehot = onehot_func( anchor2_pos_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )
            anchor2_neg2_train_onehot = onehot_func( anchor2_neg2_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )

        # combined together
        train_pos = np.dstack((anchor1_pos_train_onehot, anchor2_pos_train_onehot)) # Positive & Positive Merge horizontally dstack & Merge vertically hstack
        train_neg = np.dstack((anchor1_neg2_train_onehot, anchor2_neg2_train_onehot)) # Negative & Negative Merge horizontally dstack & Merge vertically hstack

        # 检查用
        # fancy_print('anchor1_pos_train_onehot', anchor1_pos_train_onehot)

        print('Merged block size', train_pos.shape)
        print('PNG is being generated...')


        if train_pos.shape[0]==0 or train_pos.shape[1]==0 or train_pos.shape[2]==0:
            print('Invalid empty block, skipped!')
            continue # Empty block, skip the loop



        # pip install tqdm
        # progress bar
        # import tqdm

        # Write pictures one by one
        # for j in tqdm.trange( len(train_pos), ascii=True ):
        for j in range(len(train_pos)):
            imageio.imwrite('data/' + gen_name + '/png_train/1/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(train_pos[j]))) # Must be transposed, because PNG is inverted
            imageio.imwrite('data/' + gen_name + '/png_train/0/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(train_neg[j]))) # Must be transposed, because PNG is inverted

            imageio.imwrite('data/' + gen_name + '/train_en/1/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(anchor1_pos_train_onehot[j]))) # 必须转置，因为PNG是反的
            imageio.imwrite('data/' + gen_name + '/train_en/0/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(anchor1_neg2_train_onehot[j]))) # 必须转置，因为PNG是反的
            imageio.imwrite('data/' + gen_name + '/train_pr/1/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(anchor2_pos_train_onehot[j]))) # 必须转置，因为PNG是反的
            imageio.imwrite('data/' + gen_name + '/train_pr/0/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(anchor2_neg2_train_onehot[j]))) # 必须转置，因为PNG是反的

            pic_num += 1
        
        

    ####################
    # Save the test set as picture
    ####################

    print()
    print()
    print()

    print( '>>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print('Writing test set and tags to png ...')

    anchor1_pos_test_onehot = onehot_func( anchor1_pos_test, ACGTN )
    anchor1_neg2_test_onehot = onehot_func( anchor1_neg2_test, ACGTN )
    anchor2_pos_test_onehot = onehot_func( anchor2_pos_test, ACGTN )
    anchor2_neg2_test_onehot = onehot_func( anchor2_neg2_test, ACGTN )

    # combined together
    test_pos = np.dstack((anchor1_pos_test_onehot, anchor2_pos_test_onehot)) # Positive & Positive Merge horizontally dstack & Merge vertically hstack
    test_neg = np.dstack((anchor1_neg2_test_onehot, anchor2_neg2_test_onehot)) # Negative & Negative Merge horizontally dstack & Merge vertically hstack

    print('Merged block size', test_pos.shape)
    print('PNG is being generated...')

    # Write pictures one by one
    # for j in tqdm.trange( len(test_pos), ascii=True ):
    for j in range(len(test_pos)):
        imageio.imwrite('data/' + gen_name + '/png_test/1/'+str(j)+'.png', img_as_ubyte(np.transpose(test_pos[j]))) # Must be transposed, because PNG is inverted
        imageio.imwrite('data/' + gen_name + '/png_test/0/'+str(j)+'.png', img_as_ubyte(np.transpose(test_neg[j]))) # Must be transposed, because PNG is inverted

        imageio.imwrite('data/' + gen_name + '/test_en/1/'+str(j)+'.png', img_as_ubyte(np.transpose(anchor1_pos_test_onehot[j]))) # 必须转置，因为PNG是反的
        imageio.imwrite('data/' + gen_name + '/test_en/0/'+str(j)+'.png', img_as_ubyte(np.transpose(anchor1_neg2_test_onehot[j]))) # 必须转置，因为PNG是反的
        imageio.imwrite('data/' + gen_name + '/test_pr/1/'+str(j)+'.png', img_as_ubyte(np.transpose(anchor2_pos_test_onehot[j]))) # 必须转置，因为PNG是反的
        imageio.imwrite('data/' + gen_name + '/test_pr/0/'+str(j)+'.png', img_as_ubyte(np.transpose(anchor2_neg2_test_onehot[j]))) # 必须转置，因为PNG是反的

    print( '\n>>> ' + time.asctime(time.localtime(time.time())) + '\n\n' )
    










import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 全部使用嵌套结构

# 句子拆成单词
def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq

# 单词转化为标号
def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq

# 句子转化成标号
def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq

# 对应的 ACGT 代号
def get_tokenizer():
    f = ['A', 'C', 'G', 'T']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer

# 获取数据
def get_data(enhancers, promoters):
    tokenizer = get_tokenizer()
    
    MAX_LEN = 10001
    
    X_en = sentence2num(enhancers, tokenizer, MAX_LEN)
    X_pr = sentence2num(promoters, tokenizer, MAX_LEN)

    return X_en, X_pr










########################################
#
# 文件路径系统
#
########################################
import os
 
def mkdir(path):
    if not os.path.exists(path): # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path) # makedirs 创建文件时如果路径不存在会创建这个路径
        print('-> make new folder: ' + path)
    else:
        print('-> ' + path + ' folder already exist. pass.')
		
# 使用 mkdir(相对路径/绝对路径皆可) # 调用函数



########################################
#
# main function
#
########################################

def data_prep(gen_name):

	print(gen_name)
	print()

	# 文件路径部分
	mkdir('data/' + gen_name + '/png_train/0')
	mkdir('data/' + gen_name + '/png_train/1')
	mkdir('data/' + gen_name + '/png_test/0')
	mkdir('data/' + gen_name + '/png_test/1')


	mkdir('data/' + gen_name + '/train_en/0')
	mkdir('data/' + gen_name + '/train_en/1')
	mkdir('data/' + gen_name + '/train_pr/0')
	mkdir('data/' + gen_name + '/train_pr/1')
	
	mkdir('data/' + gen_name + '/test_en/0')
	mkdir('data/' + gen_name + '/test_en/1')
	mkdir('data/' + gen_name + '/test_pr/0')
	mkdir('data/' + gen_name + '/test_pr/1')

	

	# fancy_print('merge_before_train')

	# There are many benefits of using PNG: small size, fast speed, intuitive,
	# and you can directly use keras API (high efficiency)
	print('-> processing png...')
	print()

	data_process(gen_name)
	print()
	
	print('-> png processed successfully!')
	print()

	# print('\nAll operations have been completed!')
	# input('\nPlease press any key to continue...') # Avoid crash









	print('-> processing npz...')
	print()
	
	####################
	# anchor1
	####################

	print('Loading anchor1 ...')

	f = open('data/' + gen_name + '/seq.anchor1.pos.txt', 'r')
	anchor1 = f.readlines()
	len_pos = len(anchor1)
	# fancy_print('len_pos', len_pos, '=')

	f = open('data/' + gen_name + '/seq.anchor1.neg2.txt', 'r')
	anchor1.extend(f.readlines())
	len_neg = len(anchor1) - len_pos
	# fancy_print('len_neg', len_neg, '=')

	label = [1] * len_pos + [0] * len_neg

	from sklearn.model_selection import train_test_split

	anchor1_train, anchor1_test, anchor1_label_train, anchor1_label_test = train_test_split(anchor1, label, test_size = 0.1, random_state = 42)

	# fancy_print('anchor1_train.shape', np.array(anchor1_train).shape, '+')
	# fancy_print('anchor1_test.shape', np.array(anchor1_test).shape, '+')
	# fancy_print('anchor1_label_train.shape', np.array(anchor1_label_train).shape, '+')
	# fancy_print('anchor1_label_test.shape', np.array(anchor1_label_test).shape, '+')
	print('anchor1 finished ...')
	print()

	

	####################
	# anchor2
	####################

	print('Loading anchor2 ...')

	f = open('data/' + gen_name + '/seq.anchor2.pos.txt', 'r')
	anchor2 = f.readlines()

	f = open('data/' + gen_name + '/seq.anchor2.neg2.txt', 'r')
	anchor2.extend(f.readlines())

	anchor2_train, anchor2_test, anchor2_label_train, anchor2_label_test = train_test_split(anchor2, label, test_size = 0.1, random_state = 42)

	print('anchor2 finished ...')
	print()

	

	####################
	# 填充数据
	####################

	print('train enconding ...')

	X_enhancer_train, X_promoter_train = get_data(anchor1_train, anchor2_train)

	print('test enconding ...')
	print()

	X_enhancer_test, X_promoter_test = get_data(anchor1_test, anchor2_test)

	print('writing data into .npz ...')
	print()

	

	####################
	# 写入数据
	####################

	np.savez('data/' + gen_name + '/embedding_train.npz',
			 X_en_tra = X_enhancer_train, X_pr_tra = X_promoter_train, y_tra = anchor1_label_train)

	np.savez('data/' + gen_name + '/embedding_test.npz',
			 X_en_tes = X_enhancer_test, X_pr_tes = X_promoter_test, y_tes = anchor1_label_test)
	
	print('-> npz processed successfully!')
	print('\n'*10)

	





########################################
#
# 检修区
#
########################################
if __name__ == '__main__':
    # data_prep('X5628FC')
    pass
