f = open('log_filted.txt', 'r')
lines = f.readlines()

# 保存所有拆分之后的结果
arr = []

for line in lines:
    # print(line.split())
    arr.append(line.split())

name = []

for i in arr:
    name.append(i[0])

name = list(set(name))
# print(name)



data_minmax = []
for i in arr:
    data_minmax.append(float(i[3]))
    
print(min(data_minmax))
print(max(data_minmax))



flag = 0



# 所有基因的名字
for i in name:

    onehot_cnn_one_branch = []
    onehot_cnn_two_branch = []
    onehot_embedding_dense = []
    onehot_dense = []
    onehot_resnet18 = []
    # onehot_resnet34 = []
    
    embedding_cnn_one_branch = []
    embedding_cnn_two_branch = []
    embedding_dense = []
    
    onehot_embedding_cnn_one_branch = []
    onehot_embedding_cnn_two_branch = []
    
    name_list = []
    num_list = []

    for j in arr:
        # j[0] 训练基因名称 j[1] 目标基因名称 j[2] 模型名称 j[3] 结果
        if j[0] == i and j[1] != i:

            if j[2] == 'onehot_cnn_one_branch': onehot_cnn_one_branch.append(float(j[3]))
            if j[2] == 'onehot_cnn_two_branch': onehot_cnn_two_branch.append(float(j[3]))
            if j[2] == 'onehot_embedding_dense': onehot_embedding_dense.append(float(j[3]))
            if j[2] == 'onehot_dense': onehot_dense.append(float(j[3]))
            if j[2] == 'onehot_resnet18': onehot_resnet18.append(float(j[3]))
            # if j[2] == 'onehot_resnet34': onehot_resnet34.append(float(j[3]))

            if j[2] == 'embedding_cnn_one_branch': embedding_cnn_one_branch.append(float(j[3]))
            if j[2] == 'embedding_cnn_two_branch': embedding_cnn_two_branch.append(float(j[3]))
            if j[2] == 'embedding_dense': embedding_dense.append(float(j[3]))

            if j[2] == 'onehot_embedding_cnn_one_branch': onehot_embedding_cnn_one_branch.append(float(j[3]))
            if j[2] == 'onehot_embedding_cnn_two_branch': onehot_embedding_cnn_two_branch.append(float(j[3]))
      


    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(7, 10))
    plt.gcf().subplots_adjust(bottom = 0.3)

    import pylab as pl
    pl.xticks(rotation = 90)

    plt.title(i)
    plt.xlabel('Model Name')
    plt.ylabel('Area Under the Curve')
    minmax = 0.02
    plt.ylim( min(data_minmax)*(1-minmax), 0.9) # max(data_minmax)*(1+minmax) )

    # plt.legend(labels = '')

    # plt.xlim(0.45, 0.9)
    
    # plt.barh(range(len(num_list)), num_list,
    #          tick_label = name_list, ) # color = plt.get_cmap('cool')(np.linspace(0, 1, 11))
    # plt.show()
    
    all_data = [
        onehot_cnn_one_branch,
        onehot_cnn_two_branch,
        onehot_embedding_dense,
        onehot_dense,
        onehot_resnet18,
        # onehot_resnet34,
        
        embedding_cnn_one_branch,
        embedding_cnn_two_branch,
        embedding_dense,
        
        onehot_embedding_cnn_one_branch,
        onehot_embedding_cnn_two_branch,
        ]
    
    all_label = [
        'onehot_cnn_one_branch',
        'onehot_cnn_two_branch',
        'onehot_embedding_dense',
        'onehot_dense',
        'onehot_resnet18',
        # 'onehot_resnet34',
        
        'embedding_cnn_one_branch',
        'embedding_cnn_two_branch',
        'embedding_dense',
        
        'onehot_embedding_cnn_one_branch',
        'onehot_embedding_cnn_two_branch',
        ]
    
    plt.boxplot( all_data,
                 labels = all_label,
                 # patch_artist = True,
                 showmeans = True,
                 sym = 'x',
                 )
    
    plt.grid() # 生成网格
    plt.savefig('fig/' + i + '.png')
    # plt.show()

    print(i)

    # 关闭当前显示的图像
    plt.close()

    # input('halt')








