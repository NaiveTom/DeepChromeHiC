gens = ['AD2.pp', 'AO.po', 'BL1.po', 'CM.po',
        'GM.po', 'GM.pp',
        'H1.po', 'HCmerge.po', 'IMR90.po', 'IMR90.pp',
        'LI11.po', 'ME.pp', 'MSC.po', 'MSC.pp',
        'NPC.po', 'SX.po', 'SX.pp',
        'TH1.po', 'X5628FC.po', 'X5628FC.pp']
# 'EG2.po', 'TB.po', 'FT2.po', 'SG1.pp', 

# print(gens)
# print(len(gens))





# prep

for i in gens:

    print('sbatch ' + 'prep_' + i + '.sh')

    f = open('prep_' + i + '.sh', 'w')
    
    f.write('#!/bin/bash'); f.write('\n')
    f.write('\n')
    f.write('#SBATCH -J prep_' + i); f.write('\n')
    f.write('#SBATCH -p general'); f.write('\n')
    f.write('#SBATCH -o log/prep_' + i + '.log'); f.write('\n') # _%j
    f.write('#SBATCH -e log/prep_' + i + '.err'); f.write('\n') # _%j
    f.write('#SBATCH --mail-type=END,FAIL'); f.write('\n')
    f.write('#SBATCH --mail-user=bizi@iu.edu'); f.write('\n')
    f.write('#SBATCH --nodes=1'); f.write('\n')
    f.write('#SBATCH --ntasks-per-node=1'); f.write('\n')
    f.write('#SBATCH --cpus-per-task=1'); f.write('\n')
    f.write('#SBATCH --time=48:00:00'); f.write('\n')
    f.write('#SBATCH --mem=64G'); f.write('\n')
    f.write('\n')

    f.write('module load deeplearning/2.3.0'); f.write('\n')
    f.write('\n')
    f.write('cd ..'); f.write('\n')
    f.write('\n')
    f.write('python3 DeepChromeHiC.py -p true -n ' + i); f.write('\n')

    f.close()





print('\n'*5)





# train

for i in gens:

    print('sbatch ' + 'train_cnn_' + i + '.sh')

    f = open('train_cnn_' + i + '.sh', 'w')
        
    f.write('#!/bin/bash'); f.write('\n')
    f.write('\n')
    f.write('#SBATCH -J train_cnn_' + i); f.write('\n')
    f.write('#SBATCH -p dl'); f.write('\n')
    f.write('#SBATCH --gres=gpu:v100:1'); f.write('\n')
    f.write('#SBATCH -o log/train_cnn_' + i + '.log'); f.write('\n') # _%j
    f.write('#SBATCH -e log/train_cnn_' + i + '.err'); f.write('\n') # _%j
    f.write('#SBATCH --mail-type=END,FAIL'); f.write('\n')
    f.write('#SBATCH --mail-user=bizi@iu.edu'); f.write('\n')
    f.write('#SBATCH --nodes=1'); f.write('\n')
    f.write('#SBATCH --ntasks-per-node=1'); f.write('\n')
    f.write('#SBATCH --cpus-per-task=1'); f.write('\n')
    f.write('#SBATCH --time=48:00:00'); f.write('\n')
    f.write('#SBATCH --mem=32G'); f.write('\n')
    f.write('\n')

    f.write('module load deeplearning/2.3.0'); f.write('\n')
    f.write('\n')
    f.write('cd ..'); f.write('\n')
    f.write('\n')


        
    f.write('python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t train -n ' + i); f.write('\n')
    
    f.write('\n')
        
    f.write('python3 DeepChromeHiC.py -m onehot_dense -t train -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_resnet18 -t train -n ' + i); f.write('\n')
    # f.write('python3 DeepChromeHiC.py -m onehot_resnet34 -t train -n ' + i); f.write('\n')

    f.close()





    print('sbatch ' + 'train_em_' + i + '.sh')

    f = open('train_em_' + i + '.sh', 'w')
        
    f.write('#!/bin/bash'); f.write('\n')
    f.write('\n')
    f.write('#SBATCH -J train_em_' + i); f.write('\n')
    f.write('#SBATCH -p dl'); f.write('\n')
    f.write('#SBATCH --gres=gpu:v100:1'); f.write('\n')
    f.write('#SBATCH -o log/train_em_' + i + '.log'); f.write('\n') # _%j
    f.write('#SBATCH -e log/train_em_' + i + '.err'); f.write('\n') # _%j
    f.write('#SBATCH --mail-type=END,FAIL'); f.write('\n')
    f.write('#SBATCH --mail-user=bizi@iu.edu'); f.write('\n')
    f.write('#SBATCH --nodes=1'); f.write('\n')
    f.write('#SBATCH --ntasks-per-node=1'); f.write('\n')
    f.write('#SBATCH --cpus-per-task=1'); f.write('\n')
    f.write('#SBATCH --time=48:00:00'); f.write('\n')
    f.write('#SBATCH --mem=32G'); f.write('\n')
    f.write('\n')

    f.write('module load deeplearning/2.3.0'); f.write('\n')
    f.write('\n')
    f.write('cd ..'); f.write('\n')
    f.write('\n')



    f.write('python3 DeepChromeHiC.py -m embedding_dense -t train -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t train -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t train -n ' + i); f.write('\n')

    f.close()





    print('sbatch ' + 'train_em2_' + i + '.sh')

    f = open('train_em2_' + i + '.sh', 'w')
        
    f.write('#!/bin/bash'); f.write('\n')
    f.write('\n')
    f.write('#SBATCH -J train_em2_' + i); f.write('\n')
    f.write('#SBATCH -p dl'); f.write('\n')
    f.write('#SBATCH --gres=gpu:v100:1'); f.write('\n')
    f.write('#SBATCH -o log/train_em2_' + i + '.log'); f.write('\n') # _%j
    f.write('#SBATCH -e log/train_em2_' + i + '.err'); f.write('\n') # _%j
    f.write('#SBATCH --mail-type=END,FAIL'); f.write('\n')
    f.write('#SBATCH --mail-user=bizi@iu.edu'); f.write('\n')
    f.write('#SBATCH --nodes=1'); f.write('\n')
    f.write('#SBATCH --ntasks-per-node=1'); f.write('\n')
    f.write('#SBATCH --cpus-per-task=1'); f.write('\n')
    f.write('#SBATCH --time=48:00:00'); f.write('\n')
    f.write('#SBATCH --mem=32G'); f.write('\n')
    f.write('\n')

    f.write('module load deeplearning/2.3.0'); f.write('\n')
    f.write('\n')
    f.write('cd ..'); f.write('\n')
    f.write('\n')


        
    f.write('python3 DeepChromeHiC.py -m onehot_embedding_dense -t train -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t train -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t train -n ' + i); f.write('\n')
        
    f.close()
    




print('\n'*5)



"""

# test

print('sbatch ' + 'test.sh')

f = open('test.sh', 'w')
    
f.write('#!/bin/bash'); f.write('\n')
f.write('\n')
f.write('#SBATCH -J test'); f.write('\n')
f.write('#SBATCH -p dl'); f.write('\n')
f.write('#SBATCH --gres=gpu:v100:1'); f.write('\n')
f.write('#SBATCH -o log/test_%j.log'); f.write('\n')
f.write('#SBATCH -e log/test_%j.err'); f.write('\n')
f.write('#SBATCH --mail-type=ALL'); f.write('\n')
f.write('#SBATCH --mail-user=bizi@iu.edu'); f.write('\n')
f.write('#SBATCH --nodes=1'); f.write('\n')
f.write('#SBATCH --ntasks-per-node=1'); f.write('\n')
f.write('#SBATCH --time=48:00:00'); f.write('\n')
f.write('\n')

f.write('module load deeplearning/2.3.0'); f.write('\n')
f.write('\n')
f.write('cd ..'); f.write('\n')

for i in gens:

    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write('\n')

    f.write('python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n ' + i); f.write('\n')
    f.write('\n')
    
    f.write('python3 DeepChromeHiC.py -m onehot_dense -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n ' + i); f.write('\n')
    f.write('\n')
    
    f.write('python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m embedding_dense -t test -n ' + i); f.write('\n')
    f.write('\n')
    
    f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n ' + i); f.write('\n')
    f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n ' + i); f.write('\n')
    

f.close()

"""
