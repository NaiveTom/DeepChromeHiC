#!/bin/bash

#SBATCH -J prep_SX.pp
#SBATCH -p general
#SBATCH -o log/prep_SX.pp.log
#SBATCH -e log/prep_SX.pp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G

module load deeplearning/2.3.0

cd ..

python3 DeepChromeHiC.py -p true -n SX.pp
