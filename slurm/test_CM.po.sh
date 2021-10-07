#!/bin/bash

#SBATCH -J test_CM.po
#SBATCH -p dl
#SBATCH --gres=gpu:v100:1
#SBATCH -o log/test_CM.po.log
#SBATCH -e log/test_CM.po.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load deeplearning/2.3.0

cd ..





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o NPC.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o NPC.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o NPC.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o NPC.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o NPC.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o NPC.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o NPC.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o NPC.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o NPC.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o NPC.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o GM.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o GM.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o GM.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o GM.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o GM.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o GM.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o GM.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o GM.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o GM.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o GM.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o BL1.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o BL1.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o BL1.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o BL1.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o BL1.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o BL1.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o BL1.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o BL1.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o BL1.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o BL1.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o AO.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o AO.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o AO.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o AO.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o AO.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o AO.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o AO.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o AO.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o AO.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o AO.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o LI11.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o LI11.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o LI11.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o LI11.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o LI11.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o LI11.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o LI11.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o LI11.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o LI11.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o LI11.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o CM.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o CM.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o CM.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o CM.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o CM.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o CM.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o CM.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o CM.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o CM.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o CM.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o H1.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o H1.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o H1.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o H1.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o H1.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o H1.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o H1.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o H1.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o H1.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o H1.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o TH1.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o TH1.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o TH1.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o TH1.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o TH1.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o TH1.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o TH1.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o TH1.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o TH1.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o TH1.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o IMR90.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o IMR90.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o IMR90.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o IMR90.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o IMR90.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o IMR90.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o IMR90.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o IMR90.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o IMR90.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o IMR90.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o MSC.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o MSC.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o MSC.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o MSC.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o MSC.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o MSC.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o MSC.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o MSC.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o MSC.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o MSC.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o HCmerge.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o HCmerge.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o HCmerge.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o HCmerge.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o HCmerge.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o HCmerge.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o HCmerge.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o HCmerge.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o HCmerge.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o HCmerge.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o SX.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o SX.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o SX.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o SX.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o SX.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o SX.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o SX.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o SX.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o SX.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o SX.po





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n CM.po -o X5628FC.po
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n CM.po -o X5628FC.po
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n CM.po -o X5628FC.po

python3 DeepChromeHiC.py -m onehot_dense -t test -n CM.po -o X5628FC.po
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n CM.po -o X5628FC.po

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n CM.po -o X5628FC.po
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n CM.po -o X5628FC.po
python3 DeepChromeHiC.py -m embedding_dense -t test -n CM.po -o X5628FC.po

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n CM.po -o X5628FC.po
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n CM.po -o X5628FC.po
