#!/bin/bash
#SBATCH -J 1_Faceskin_32_Run
#SBATCH -p a100_normal_q
#SBATCH -N 2 --ntasks-per-node=1 --cpus-per-task=32 # 16 cpus/gpu recommended, 16GB/cpu memory automatically provided
#SBATCH -t 2-00:00:00 # d-hh:mm:ss
#SBATCH --gres=gpu:2 #how many gpus on each node
#SBATCH --account=abbott
#SBATCH --mail-user=yogeshd@vt.edu
#SBATCH --mail-type=BEGIN,END,ABORT
#SBATCH --export=NONE # this makes sure the compute environment is clean
#SBATCH --output=/home/yogeshd/Skin/slurm-%j.out

#load module
module load Anaconda3/2020.11
module load cuDNN/8.1.1.33-CUDA-11.2.1
source activate yoz

#-cp /projects/abbott/images.tar $TMPNVME
# change to local NVME disk and extract the dataset
#-cd $TMPNVME
#-tar -xf $TMPNVME/images.tar
#-export DATADIR=$TMPNVME/
# move back to working directory
#-cd $SLURM_SUBMIT_DIR

#copy tarfile from projects to local NVME disk
#-cp /projects/abbott/aligned_img.tar $TMPNVME
# change to local NVME disk and extract the dataset
#-cd $TMPNVME
#-tar -xf $TMPNVME/aligned_img.tar
#-export DATADIR=$TMPNVME/
# move back to working directory
#-cd $SLURM_SUBMIT_DIR

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
export MASTER_PORT=12698
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

#-p -> Path to the Face_Frames.csv
#-s -> Path to the Skin_Frames.csv
#-lr -> Learning Rate
#-e -> Number of epochs
#-opt -> Optimizer
#srun --export=ALL python Code_Filename.py -p './Face_Frame_csvFilename.csv' -s './Skin_Frame_csvFilename.csv' -m 'Model_checkpoint_saving_name.pt' -lr 0.0001 -e 500 -opt 'SGD'

####srun --export=ALL python TrnTstFaceSkin.py -p'./myoutface_37P2minto1min_filteredclean.csv' -s './myoutface_37P2minto1min_filteredclean.csv' -m 'myfiltclean37P_2minto1min_CorrectedFaceFacenormsbest_corrstdbvp_32_492real72_Dmetanh_dp50_dpe50_288e_0.0001l_128bs_sgdwtd0.0001_bvp_cleanskin.pt' -lr 0.0001 -e 288 -opt 'SGD'
srun --export=ALL python TrnTstFaceSkin.py -p'./myout_cohfaceface_37P2minto1min_clean.csv' -s './myout_cohfaceskin_37P2minto1min_clean.csv' -m 'test1_cohface_clean37P_2minto1min_SkinFacenormsbest_corrstdbvp_32_492real72_Dmetanh_dp50_dpe50_500e_0.0001l_128bs_sgdwtd0.0001_bvp.pt' -lr 0.0001 -e 500 -opt 'SGD'