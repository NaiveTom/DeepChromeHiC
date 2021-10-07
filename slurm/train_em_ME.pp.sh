#!/bin/bash

#SBATCH -J train_em_ME.pp
#SBATCH -p dl
#SBATCH --gres=gpu:v100:1
#SBATCH -o log/train_em_ME.pp.log
#SBATCH -e log/train_em_ME.pp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load deeplearning/2.3.0

cd ..

python3 DeepChromeHiC.py -m embedding_dense -t train -n ME.pp
python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t train -n ME.pp
python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t train -n ME.pp
