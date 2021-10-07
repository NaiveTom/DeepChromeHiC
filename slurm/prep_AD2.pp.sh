#!/bin/bash

#SBATCH -J prep_AD2.pp
#SBATCH -p general
#SBATCH -o log/prep_AD2.pp.log
#SBATCH -e log/prep_AD2.pp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bizi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G

module load deeplearning/2.3.0

cd ..

python3 DeepChromeHiC.py -p true -n AD2.pp
