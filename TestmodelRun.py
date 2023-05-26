## Author: Yogesh Deshpande Aug 2021 - May 2023

################ Imports ###############
import numpy as np
import pandas as pd
import os
import random
import plotly
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms

from sklearn.model_selection import train_test_split
from skimage import io
from PIL import Image

import argparse
import builtins

# Path, Best chk model name, Learning rate, Batch Size, No. of Epochs
parser = argparse.ArgumentParser(description='Input Essentials')
parser.add_argument('-p', '--path', type=str, metavar='', required=True, help='Path to the output csv file')
parser.add_argument('-s', '--skinpath', type=str, metavar='', required=True, help='Path to the skin output csv file')
parser.add_argument('-m', '--modelname', type=str, metavar='', required=True, help='Best checkpoint model name')
parser.add_argument('-opt', '--optimizer', type=str, metavar='', required=True, help='Adam or Adadelta')
parser.add_argument('-lr', '--learningrate', type=float, metavar='', required=True, help='Learning rate')
parser.add_argument('-e', '--epochs', type=int, metavar='', required=True, help='No. of Epochs')

# DDP configs:
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

args = parser.parse_args()

########################################################################################################
# DDP setting
if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ['SLURM_NPROCS'])
    # args.world_size = int(os.environ["WORLD_SIZE"])
args.distributed = args.world_size > 1
ngpus_per_node = torch.cuda.device_count()
if args.distributed:
    if 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    tensor_list = []
    for dev_idx in range(torch.cuda.device_count()):
        tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
    
    dist.all_reduce_multigpu(tensor_list)

# suppress printing if not on master gpu
if args.rank!=0:
    def print_pass(*args):
        pass
    builtins.print = print_pass
########################################################################################################
#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
##torch.backends.cudnn.benchmark = True

############################### Function For Processing ##################################
class get_data(Dataset):
    def __init__(self, df, dfs):
        self.df = df
        self.dfs = dfs
        self.transform = transforms.Compose([
            transforms.Resize((72, 72)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        '''
        Description for reference
        model1 -> Is the model with BVP first derivative signal as the ground truth
        model2 -> Is the model with original BVP signal as the ground truth
        '''

        bvpcur = self.df['BVP Values'][i] ## Comment when working with model2
        ##bvp = self.df['BVP Values'][i] ## Uncomment when working with model2

        ims = os.environ.get('DATADIR')+ self.dfs['Filepath'][i][2:]
        ims = Image.open(ims).convert('RGB')
        ims = self.transform(ims)

        #im = "./" + self.df['Filepath'][i]
        im = os.environ.get('DATADIR')+ self.df['Filepath'][i][2:]
        im = Image.open(im).convert('RGB')
        im = self.transform(im)

        try:
            if self.df['Filepath'][i].split('/')[2] == self.df['Filepath'][i + 1].split('/')[2]:
                im_next = os.environ.get('DATADIR') + self.df['Filepath'][i + 1][2:]
                bvp_next = self.df['BVP Values'][i + 1] ## Comment when working with model2
                
                # im_next_nxt = os.environ.get('DATADIR') + self.df['Filepath'][i + 2][2:]
                # bvp_next_nxt = self.df['BVP Values'][i + 2]
            else:
                im_next = os.environ.get('DATADIR') + self.df['Filepath'][i - 1][2:]
                bvp_next = self.df['BVP Values'][i - 1] ## Comment when working with model2
                
                # im_next_nxt = os.environ.get('DATADIR') + self.df['Filepath'][i - 2][2:]
                # bvp_next_nxt = self.df['BVP Values'][i - 2]
        except:
            im_next = os.environ.get('DATADIR') + self.df['Filepath'][i - 1][2:]
            bvp_next = self.df['BVP Values'][i - 1] ## Comment when working with model2
            
            # im_next_nxt = os.environ.get('DATADIR') + self.df['Filepath'][i - 2][2:]
            # bvp_next_nxt = self.df['BVP Values'][i - 2]
        
        im_next = Image.open(im_next).convert('RGB')
        im_next = self.transform(im_next)
        im_norm = (im_next - im) / (im_next + im)
        ##im_norm = (im_next_nxt - im_next) - (im_next - im)
        
        ####
        std, mean = torch.std_mean(im_norm, dim=[1, 2])
        thrdstdmax = mean+(3*std)
        thrdstdmin = mean-(3*std)
        im_norm[0] = torch.clip(im_norm[0], max = thrdstdmax[0])
        im_norm[1] = torch.clip(im_norm[1], max = thrdstdmax[1])
        im_norm[2] = torch.clip(im_norm[2], max = thrdstdmax[2])
        im_norm = torch.nan_to_num(im_norm)

        bvp = bvp_next - bvpcur ## Comment when working with model2
        ##bvp = (bvp_next_nxt - bvp_next) - (bvp_next - bvpcur) ## I think it was for second derivative
        bvp = torch.tensor(bvp)

        std, mean = torch.std_mean(ims, dim=[1, 2])
        std = 0.0000001 if (std == 0).any() else std
        normyo = transforms.Normalize(mean, std)
        ims = normyo(ims)
        ####
        return (ims, im_norm, bvp)

################################# Call the csv file #############################################

df = pd.read_csv(args.path)#, nrows=97355)  #nrows=118190, 244480 // clean 97355, 141195, 198455 // cohface = 103065 // cohface clean 96545
dfs = pd.read_csv(args.skinpath)#, nrows=97355)

################################################################################################

dataset = get_data(df, dfs)

# train, test = train_test_split(df, test_size=0.2, random_state=129)
##train_val, test = train_test_split(dataset, test_size=0.2, shuffle=False)
##train, val = train_test_split(train_val, test_size=0.25, shuffle=False)  # 0.25 x 0.8 = 0.2
train, test = train_test_split(dataset, test_size=0.25, shuffle=False)

print(f'- Number of Datapoints in Training Set: {len(train)}')
print(f'- Number of Datapoints in Validation Set: {len(test)}')
print(f'- Number of Datapoints in Test Set: {len(test)}')
####################################### Batch Size Handle ########################################
SEED = 1

# CUDA?
use_cuda = torch.cuda.is_available()
print("CUDA Available:", use_cuda)

# For reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if use_cuda:
    torch.cuda.manual_seed_all(SEED)
    BATCH_SIZE = 128
else:
    BATCH_SIZE = 32

print('BATCH_SIZE:', BATCH_SIZE)

###################################### Data Loader ########################################
#kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
##train_sampler = DistributedSampler(train, shuffle=False)
##train_loader = DataLoader(dataset = train, batch_size=BATCH_SIZE//args.world_size, pin_memory = True, sampler=train_sampler, shuffle=False)
train_loader = DataLoader(dataset = train, batch_size=BATCH_SIZE//args.world_size, pin_memory = True, shuffle=False)
##val_loader = DataLoader(dataset = val, batch_size=BATCH_SIZE//args.world_size, pin_memory = True, drop_last=True, shuffle=False)
test_loader = DataLoader(dataset = test, batch_size=BATCH_SIZE//args.world_size, pin_memory = True, shuffle=False)
###########################################################################################

import Deepyo

model = Deepyo.Yomodel()
model_name = 'myfiltclean37P_2minto1min_CorrectedSkinFacenormsbest_corrstdbvp_32_492real72_Dmetanh_dp50_dpe50_325e_0.0001l_128bs_sgdwtd0.0001_bvp_cleanskin'
model.load_state_dict(torch.load('./'+model_name+'.pt', map_location=torch.device('cuda')))
##model = nn.DataParallel(model)
##model = model.to(device)

if args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        ##model_without_ddp = model.module
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        ##model_without_ddp = model.module
else:
    raise NotImplementedError("Only DistributedDataParallel is supported.")

#loss and optimizer
def criterion_pear(outcome, aim):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    pearson = cos(outcome - outcome.mean(dim=0, keepdim=True), aim - aim.mean(dim=0, keepdim=True))
    return 1-pearson
criterion = nn.MSELoss().to(device)
criterion_mae = nn.L1Loss().to(device)
#optimizer = optim.Adam(model.parameters(), lr = 0.01)
if args.optimizer == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr = args.learningrate)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr = args.learningrate, weight_decay=0.0005)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr = args.learningrate, momentum=0.9, weight_decay=0.0001)

##scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000], gamma=0.1)

num_epochs = args.epochs
best_loss = 1000000.0
dataprev = []

######################################### Test and Evaluate ####################################

def check_accuracy(loader, model, s):
    model.eval()
    losses = []
    maelosses = []
    output_bvp =[]
    label_bvp = []
    pear_losses = []

    with torch.no_grad():
        for x, z, y in loader:
            x = x.to(device=device, non_blocking=True)
            z = z.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            y = y.float()
            
            ## bvp_std ##
            std_bvp, mean_bvp = torch.std_mean(y)
            y = (y-mean_bvp)/std_bvp
            ##

            scores = model(x, z).reshape(-1)
            # _, predictions = scores.max(1)
            # num_correct += (predictions == y).sum()
            # num_samples += predictions.size(0)
            loss = criterion(scores, y)
            losses.append(loss.item())
            loss1 = criterion_mae(scores, y)
            maelosses.append(loss1.item())
            loss_pear = criterion_pear(scores, y)
            pear_losses.append(loss_pear.item())
            output_bvp.append(scores)
            label_bvp.append(y)

        # print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 1}')
        test_loss = sum(losses) / len(losses)
        print(f'MSE Loss is {test_loss}')

        mae_loss = sum(maelosses) / len(maelosses)
        print(f'MAE Loss is {mae_loss}')
        
        pear_loss = 1 - (sum(pear_losses) / len(pear_losses))
        print(f'Pearsons Coeff is {pear_loss}')

        print(f'Output BVP is \n {output_bvp}')
        print(f'Output GT is \n {label_bvp}')
        
        BVP = []
        BVP_GT_graph = []
        for i in output_bvp:
            i = i.cpu().numpy()
            BVP.extend(i)
            #print(BVP)
        for i in label_bvp:
            i = i.cpu().numpy()
            BVP_GT_graph.extend(i)
            #print(BVP_GT_graph)
        
        ##print(len(losses), len(maelosses), len(BVP), len(BVP_GT_graph), len(output_bvp), len(label_bvp))
        data = [go.Scatter(x = np.arange(0, len(BVP)), y = BVP, name="BVP 1stD Output"), go.Scatter(x = np.arange(0, len(BVP_GT_graph)), y = BVP_GT_graph, name="BVP 1stD GT")]
        fig = go.Figure(data)
        #fig.update_xaxes(title_text='Time (sec)')
        #fig.update_yaxes(title_text='BVP First Derivative (mV)')
        fig.update_layout( title='BVP 1st Derivative Outcome', xaxis_title='Samples', yaxis_title='BVP First Derivative (mV)')
        #fig.show()
        
        plotly.offline.plot(fig, filename="./"+s+".html")
        
        with open("./"+s+".npy", 'wb') as f:
            np.save(f, np.array(BVP_GT_graph))
            np.save(f, np.array(BVP))
        
        introp = []
        intrgt = []
        for i in range(len(BVP)):
            introp.append(np.trapz(np.diff(BVP[i:i+8]), dx=-1))
        for i in range(len(BVP_GT_graph)):
            intrgt.append(np.trapz(np.diff(BVP_GT_graph[i:i+8]), dx=-1))
        
        data1 = [go.Scatter(y = introp, name="BVP Output"), go.Scatter(y = intrgt, name="BVP GT")]
        fig1 = go.Figure(data1)
        fig1.update_layout( title='BVP Der. Integration Outcome', xaxis_title='Samples', yaxis_title='BVP (mV)')
        
        plotly.offline.plot(fig1, filename="./"+s+"_inrgtBVP.html")
        
        with open("./"+s+"_intrgtBVP.npy", 'wb') as f:
            np.save(f, np.array(intrgt))
            np.save(f, np.array(introp))
    model.train()
    
    


##print("During Training Best Loss Computed Was: ", best_loss)
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model, 'crossdataval_uonc_trainoutput'+model_name)

print("Checking accuracy on Val Set")
check_accuracy(test_loader, model, 'crossdataval_uonc_testoutput'+model_name)
