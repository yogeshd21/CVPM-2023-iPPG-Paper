#!/bin/bash
#SBATCH -J Test_frames
#SBATCH -p p100_normal_q
#SBATCH -N 2 --ntasks-per-node=1 --cpus-per-task=16 # 16 cpus/gpu recommended, 16GB/cpu memory automatically provided
#SBATCH -t 3-00:00:00 # d-hh:mm:ss
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

# cp /projects/abbott_lab/BVP/BVP/Frames.tar $TMPNVME
# #cp /projects/abbott/image.tar
# #cp /projects/abbott/aligned_image.tar
# # change to local NVME disk and extract the dataset
# cd $TMPNVME
# tar -xf $TMPNVME/Frames.tar
# export DATADIR=$TMPNVME/
# # move back to working directory
# cd $SLURM_SUBMIT_DIR

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
export MASTER_PORT=21621
### change WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=4
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
### export NCCL_IB_DISABLE=1
### export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp"
export NCCL_DEBUG=INFO

export DATADIR=$SLURM_SUBMIT_DIR/

#-p -> Path to the Video_Frames.csv
#-lr -> Learning Rate
#-e -> Number of epochs
#-opt -> Optimizer
#srun --export=ALL python Code_Filename.py -p './Video_Frame_csvFilename.csv' -m 'Model_checkpoint_saving_name.pt' -lr 1.0 -e 48 -opt 'Adadelta'

####srun --export=ALL python Testmodel.py -p './myoutframesinface_37P2minto1min_filteredclean_derstd.csv' -m 'testmyfiltclean_derstd_37P_2minto1min_CorrectedFramesinfacenormsbest_allcorrboth_e48_1.0lr_adel_bvp_Itrbicu.pt' -lr 1.0 -e 48 -opt 'Adadelta'
srun --export=ALL python Testmodel.py -p './myoutcohfaceframesinface_3min_2minto1min_clean_derstd.csv' -m 'testcohface_clean_derstd_31P_3min_2minto1min_CorrectedFramesinfacenormsbest_allcorrboth_e48_1.0lr_adel_bvp_Itrbicu.pt' -lr 1.0 -e 48 -opt 'Adadelta'