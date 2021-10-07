gens = ['AD2.pp', 'AO.po', 'BL1.po', 'CM.po',
        'GM.po', 'GM.pp',
        'H1.po', 'HCmerge.po', 'IMR90.po', 'IMR90.pp',
        'LI11.po', 'ME.pp', 'MSC.po', 'MSC.pp',
        'NPC.po', 'SX.po', 'SX.pp',
        'TH1.po', 'X5628FC.po', 'X5628FC.pp']
# 'EG2.po', 'TB.po', 'FT2.po', 'SG1.pp', 

po = [
'NPC.po',
'GM.po',
'BL1.po',
'AO.po',
'LI11.po',
'CM.po',
'H1.po',
'TH1.po',
'IMR90.po',
'MSC.po',
'HCmerge.po',
'SX.po',
'X5628FC.po',
# 'EG2.po',
# 'TB.po',
# 'FT2.po',
]

pp = [
'ME.pp',
'IMR90.pp',
'MSC.pp',
'AD2.pp',
'GM.pp',
'SX.pp',
'X5628FC.pp',
# 'SG1.pp',
]



# test

for i in gens:

    print('sbatch ' + 'test_' + i + '.sh')

    f = open('test_' + i + '.sh', 'w')
        
    f.write('#!/bin/bash'); f.write('\n')
    f.write('\n')
    f.write('#SBATCH -J test_' + i); f.write('\n')
    f.write('#SBATCH -p dl'); f.write('\n')
    f.write('#SBATCH --gres=gpu:v100:1'); f.write('\n')
    f.write('#SBATCH -o log/test_' + i + '.log'); f.write('\n') # _%j
    f.write('#SBATCH -e log/test_' + i + '.err'); f.write('\n') # _%j
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



    if i in po:
        for j in po:

            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')

            f.write('python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('\n')
                
            f.write('python3 DeepChromeHiC.py -m onehot_dense -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n ' + i + ' -o ' + j); f.write('\n')
            # f.write('python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('\n')
                
            f.write('python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m embedding_dense -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('\n')
                
            f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n ' + i + ' -o ' + j); f.write('\n')

    if i in pp:
        for j in pp:

            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')

            f.write('python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('\n')
                
            f.write('python3 DeepChromeHiC.py -m onehot_dense -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n ' + i + ' -o ' + j); f.write('\n')
            # f.write('python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('\n')
                
            f.write('python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m embedding_dense -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('\n')
                
            f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
            f.write('python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n ' + i + ' -o ' + j); f.write('\n')
        
    f.close()
