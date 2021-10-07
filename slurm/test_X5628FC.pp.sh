#!/bin/bash

#SBATCH -J test_X5628FC.pp
#SBATCH -p dl
#SBATCH --gres=gpu:v100:1
#SBATCH -o log/test_X5628FC.pp.log
#SBATCH -e log/test_X5628FC.pp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load deeplearning/2.3.0

cd ..





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o ME.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o ME.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o ME.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o ME.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o ME.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o ME.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o ME.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o ME.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o ME.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o ME.pp





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o IMR90.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o IMR90.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o IMR90.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o IMR90.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o IMR90.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o IMR90.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o IMR90.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o IMR90.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o IMR90.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o IMR90.pp





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o MSC.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o MSC.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o MSC.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o MSC.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o MSC.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o MSC.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o MSC.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o MSC.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o MSC.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o MSC.pp





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o AD2.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o AD2.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o AD2.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o AD2.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o AD2.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o AD2.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o AD2.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o AD2.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o AD2.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o AD2.pp





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o GM.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o GM.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o GM.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o GM.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o GM.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o GM.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o GM.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o GM.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o GM.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o GM.pp





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o SX.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o SX.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o SX.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o SX.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o SX.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o SX.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o SX.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o SX.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o SX.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o SX.pp





python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n X5628FC.pp -o X5628FC.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n X5628FC.pp -o X5628FC.pp
python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n X5628FC.pp -o X5628FC.pp

python3 DeepChromeHiC.py -m onehot_dense -t test -n X5628FC.pp -o X5628FC.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n X5628FC.pp -o X5628FC.pp

python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n X5628FC.pp -o X5628FC.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n X5628FC.pp -o X5628FC.pp
python3 DeepChromeHiC.py -m embedding_dense -t test -n X5628FC.pp -o X5628FC.pp

python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n X5628FC.pp -o X5628FC.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n X5628FC.pp -o X5628FC.pp
