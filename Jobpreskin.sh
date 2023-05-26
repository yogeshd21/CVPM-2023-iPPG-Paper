#!/bin/bash
#SBATCH -J skin_preprocess
#SBATCH -p a100_normal_q
#SBATCH -N 2 --ntasks-per-node=1 --cpus-per-task=32 # 16 cpus/gpu recommended, 16GB/cpu memory automatically provided
#SBATCH -t 4-00:00:00 # d-hh:mm:ss
#SBATCH --gres=gpu:2 #how many gpus on each node
#SBATCH --account=abbott
#SBATCH --mail-user=yogeshd@vt.edu
#SBATCH --mail-type=BEGIN,END,ABORT
#SBATCH --export=NONE # this makes sure the compute environment is clean
#SBATCH --output=/home/yogeshd/Frames/slurm-%j.out

#load module
module load Anaconda3/2020.11
module load cuDNN/8.1.1.33-CUDA-11.2.1
source activate yoz

python prediction_new.py
