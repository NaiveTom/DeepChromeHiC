#!/bin/bash

#SBATCH -J train_em2_GM.pp
#SBATCH -p dl
#SBATCH --gres=gpu:v100:1
#SBATCH -o log/train_em2_GM.pp.log
#SBATCH -e log/train_em2_GM.pp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load deeplearning/2.3.0

cd ..

python3 DeepChromeHiC.py -m onehot_embedding_dense -t train -n GM.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t train -n GM.pp
python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t train -n GM.pp
