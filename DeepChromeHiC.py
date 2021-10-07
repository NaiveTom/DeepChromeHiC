# -*- coding:utf-8 -*-

import argparse





if __name__ == '__main__':

    # 基本说明
    print('*'*36)
    print('*'*36)
    print()
    print('    Welcome to use DeepChromeHiC')
    print()
    print('*'*36)
    print('*'*36)
    print()

    ####################
    # 解析器
    ####################
    parser = argparse.ArgumentParser()

    # 引入一个时间，避免看不到于是就写五遍
    import time
    print( 'begin time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'begin time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'begin time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'begin time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'begin time >>> ' + time.asctime(time.localtime(time.time())) + '\n\n\n' )

    # 统计时间
    start = time.time()

    # 预训练模型
    print('If it is the first time to use, please preprocess the data first.')
    print('Use the command: python3 DeepChromeHiC.py -p true -n [gene name]')
    print('For example: python3 DeepChromeHiC.py -p true -n AD2.po')
    print()

    ####################
    # 参数：预处理
    ####################
    parser.add_argument('-p', '--preprocessing',
                        help = 'Preprocess the data, if you enter [true] (case sensitive), then proceed, if no, pass this process. Note: This command only needs to be entered once.',
                        required = False)

    ####################
    # 参数：选取模型
    ####################
    parser.add_argument('-m', '--model',
                        help = 'Enter the model name which your choose: [onehot_cnn_one_branch] / [onehot_cnn_two_branch] / [onehot_embedding_dense] / [onehot_embedding_cnn_one_branch] / [onehot_embedding_cnn_two_branch] / [onehot_dense] / [onehot_resnet18] / [onehot_resnet34] / [embedding_cnn_one_branch] / [embedding_cnn_two_branch] / [embedding_dense] (all use lowercase).',
                        required = False)

    ####################
    # 参数：选择类型 训练还是测试
    ####################
    parser.add_argument('-t', '--type',
                        help = 'Please choose [train] / [test] (all use lowercase).',
                        required = False)

    ####################
    # 参数：选取的基因名称
    ####################
    parser.add_argument('-n', '--name',
                        help = 'Enter the gene name of your choice (note: case sensitive).',
                        required = False)

    ####################
    # 参数：验证的基因
    ####################
    parser.add_argument('-o', '--object',
                        help = 'Enter the gene name of your choice (note: case sensitive).',
                        required = False)

    ####################
    # 参数：基因长度
    ####################
    parser.add_argument('-l', '--length',
                        default = 10001,
                        help = 'Enter the length of gene, default is 10001.',
                        required = False)

    args = parser.parse_args()

    print('\n=== Below is your input ===\n')
    if args.preprocessing == 'true':
        print('args.preprocessing = '+ args.preprocessing)
        print('args.name = '+ args.name)
    else:
        print('args.model = '+ args.model)
        print('args.type = '+ args.type)
        print('args.name = '+ args.name)
        print('args.length = '+ str(args.length))
        try:
            print('args.object = '+ args.object)
        except:
            pass
    print('===========================')
    print()

    # 数据预处理
    if args.preprocessing == 'true':
        
        from data_preprocessing import data_prep
        data_prep(args.name)
        
    else:

        from train import train_cnn_dense_resnet, train_cnn_separate, train_embedding
        from test import test_cnn_dense_resnet, test_cnn_separate, test_embedding



        ####################
        # 输入错误提示
        ####################
        if args.model not in ['onehot_cnn_one_branch', 'onehot_cnn_two_branch', 'onehot_embedding_dense', 
                              'onehot_dense', 'onehot_resnet18', 'onehot_resnet34',
                              'embedding_cnn_one_branch', 'embedding_cnn_two_branch', 'embedding_dense',
                              'onehot_embedding_cnn_one_branch', 'onehot_embedding_cnn_two_branch']:
            print("Wrong model name!\nUse command python3 DeepChromeHiC.py -h to see the correct model name!")
            exit(1)
            
        if args.type not in ['train', 'test']:
            print("Wrong type name!\nType must in ['train', 'test']")
            exit(1)
            
        print() # 打印一个空行





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


        # 创建路径
        mkdir('h5_weights/' + args.name) # 只需要基因名称就行了

        mkdir('result/' + args.name + '/onehot_cnn_one_branch')
        mkdir('result/' + args.name + '/onehot_cnn_two_branch')
        mkdir('result/' + args.name + '/onehot_embedding_dense')

        mkdir('result/' + args.name + '/onehot_dense')
        mkdir('result/' + args.name + '/onehot_resnet18')
        mkdir('result/' + args.name + '/onehot_resnet34')

            
        mkdir('result/' + args.name + '/embedding_cnn_one_branch')
        mkdir('result/' + args.name + '/embedding_cnn_two_branch')
        mkdir('result/' + args.name + '/embedding_dense')

        mkdir('result/' + args.name + '/onehot_embedding_cnn_one_branch')
        mkdir('result/' + args.name + '/onehot_embedding_cnn_two_branch')



        if args.model=='onehot_cnn_one_branch' or args.model=='onehot_embedding_dense' \
           or args.model=='onehot_dense' or args.model=='onehot_resnet18' or args.model=='onehot_resnet34':
   
            if args.type=='train':
                train_cnn_dense_resnet(args.name, args.model, int(args.length))
            elif args.type=='test':
                test_cnn_dense_resnet(args.name, args.model, args.object, int(args.length))



        if args.model == 'onehot_cnn_two_branch':
            
            if args.type=='train':
                train_cnn_separate(args.name, args.model, int(args.length))
            elif args.type=='test':
                test_cnn_separate(args.name, args.model, args.object, int(args.length))



        if args.model=='embedding_cnn_one_branch' or args.model=='embedding_cnn_two_branch' or args.model=='embedding_dense' \
           or args.model=='onehot_embedding_cnn_one_branch' or args.model=='onehot_embedding_cnn_two_branch':

            if args.type=='train':
                train_embedding(args.name, args.model)
            elif args.type=='test':
                test_embedding(args.name, args.model, args.object)



    # 引入一个时间，避免看不到于是就写五遍
    import time
    print( 'end time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'end time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'end time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'end time >>> ' + time.asctime(time.localtime(time.time())) + '\n' )
    print( 'end time >>> ' + time.asctime(time.localtime(time.time())) + '\n\n\n\n\n\n\n\n\n\n' )

    end = time.time()
    print()
    print()
    try:
        print('args.model = '+ args.model)
    except:
        pass
    print('time used = ' + str(end-start))
    print()
    print()






    ####################
    # 避免闪退，测试用
    ####################
    # import os
    # os.system('pause')
