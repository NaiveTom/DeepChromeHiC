#!/bin/bash

#SBATCH -J train_cnn_MSC.pp
#SBATCH -p dl
#SBATCH --gres=gpu:v100:1
#SBATCH -o log/train_cnn_MSC.pp.log
#SBATCH -e log/train_cnn_MSC.pp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load deeplearning/2.3.0

cd ..

python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n MSC.pp
python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t train -n MSC.pp

python3 DeepChromeHiC.py -m onehot_dense -t train -n MSC.pp
python3 DeepChromeHiC.py -m onehot_resnet18 -t train -n MSC.pp
