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



onehot_cnn_one_branch = []
onehot_cnn_two_branch = []
onehot_embedding_dense = []
onehot_dense = []
onehot_resnet18 = []
    
embedding_cnn_one_branch = []
embedding_cnn_two_branch = []
embedding_dense = []
    
onehot_embedding_cnn_one_branch = []
onehot_embedding_cnn_two_branch = []


# 所有基因的名字
for i in name:

    for j in arr:
        # j[0] 训练基因名称 j[1] 目标基因名称 j[2] 模型名称 j[3] 结果
        if j[0] == i and j[1] == i:

            # print(i)

            if j[2] == 'onehot_cnn_one_branch':
                onehot_cnn_one_branch.append(j[3])
                # print(i)
                # print(j[3])
            if j[2] == 'onehot_cnn_two_branch': onehot_cnn_two_branch.append(j[3])
            if j[2] == 'onehot_embedding_dense': onehot_embedding_dense.append(j[3])
            if j[2] == 'onehot_dense': onehot_dense.append(j[3])
            if j[2] == 'onehot_resnet18': onehot_resnet18.append(j[3])

            if j[2] == 'embedding_cnn_one_branch': embedding_cnn_one_branch.append(j[3])
            if j[2] == 'embedding_cnn_two_branch': embedding_cnn_two_branch.append(j[3])
            if j[2] == 'embedding_dense': embedding_dense.append(j[3])

            if j[2] == 'onehot_embedding_cnn_one_branch': onehot_embedding_cnn_one_branch.append(j[3])
            if j[2] == 'onehot_embedding_cnn_two_branch': onehot_embedding_cnn_two_branch.append(j[3])



f = open('test.csv', 'w')

f.write(',')
for i in name:
    f.write(i); f.write(',')
f.write('\n')
    
f.write('onehot_cnn_one_branch'); f.write(',')
for i in onehot_cnn_one_branch:
    print(i)
    f.write(i); f.write(',')
f.write('\n')

f.write('onehot_cnn_two_branch'); f.write(',')
for i in onehot_cnn_two_branch:
    f.write(i); f.write(',')
f.write('\n')

f.write('onehot_embedding_dense'); f.write(',')
for i in onehot_embedding_dense:
    f.write(i); f.write(',')
f.write('\n')

f.write('onehot_dense'); f.write(',')
for i in onehot_dense:
    f.write(i); f.write(',')
f.write('\n')

f.write('onehot_resnet18'); f.write(',')
for i in onehot_resnet18:
    f.write(i); f.write(',')
f.write('\n')

f.write('embedding_cnn_one_branch'); f.write(',')
for i in embedding_cnn_one_branch:
    f.write(i); f.write(',')
f.write('\n')

f.write('embedding_cnn_two_branch'); f.write(',')
for i in embedding_cnn_two_branch:
    f.write(i); f.write(',')
f.write('\n')

f.write('embedding_dense'); f.write(',')
for i in embedding_dense:
    f.write(i); f.write(',')
f.write('\n')

f.write('onehot_embedding_cnn_one_branch'); f.write(',')
for i in onehot_embedding_cnn_one_branch:
    f.write(i); f.write(',')
f.write('\n')

f.write('onehot_embedding_cnn_two_branch'); f.write(',')
for i in onehot_embedding_cnn_two_branch:
    f.write(i); f.write(',')
f.write('\n')

f.close()
