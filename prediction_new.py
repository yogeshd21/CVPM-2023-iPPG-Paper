## Author: Yogesh Deshpande Aug 2021 - May 2023

import numpy as np
import torch
from torchvision import transforms
import os
import cv2
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from tqdm import tqdm
from pathlib import Path
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = torch.load('./fcn_model_6.pt', map_location ='cpu')
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ =='__main__':
    save_filedr = './cohfaceimages'   #Path where you want to save Skin Frames
    foldr = './cohfacealigned_img'    #Path where you have Face Frames
    foldrls = os.listdir(foldr)
    for foldr_name in foldrls:
        img_file = os.path.join(foldr, foldr_name)
        img_names = os.listdir(img_file)

        save_file = os.path.join(save_filedr, foldr_name)
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        for img_name in tqdm(img_names):
            img = cv2.imread(os.path.join(img_file,img_name))
            size = img.shape
    
            imgA = cv2.resize(img, (160, 160))
    
            imgA = transform(imgA)
            imgA = imgA.to(device)
            imgA = imgA.unsqueeze(0)
            output = model(imgA)
            output = torch.sigmoid(output)
    
            output_np = output.cpu().detach().numpy().copy()
            output_np = np.squeeze(output_np)*255

            output_npA = output_np[0]
            output_npB = output_np[1]

            output_sigmoid = output_npA/(output_npA+output_npB)
    
            try:
                output = cv2.resize(output_sigmoid, (size[1], size[0]))
                b,g,r = cv2.split(img)
                output[output < 0.5] = 0
                output[output >= 0.5] = 1
                output = output.astype(int)
                b = np.multiply(b, output)
                g = np.multiply(g, output)
                r = np.multiply(r, output)
    
                output = cv2.merge([b,g,r])

                cv2.imwrite(os.path.join(save_file, img_name), output)

            except:
                os.remove(os.path.join(save_file,img_name))
    
    skin = pd.DataFrame(columns=['Filepath', 'BVP Values'])
    #facelst = pd.read_csv('./myoutface.csv') # For UBFC-Phys
    facelst = pd.read_csv('./myout_cohfaceface.csv') # For COHFACE
    
    j = 1
    for i in range(len(facelst)):
        if os.path.exists('./cohfaceimages'+facelst['Filepath'][i][20:]):
            skin.loc[j, 'Filepath'] = './cohfaceimages'+facelst['Filepath'][i][20:]
            skin.loc[j, 'BVP Values'] = facelst['BVP Values'][i]
            j+=1
    
    skin.to_csv('./myout_cohfaceskin.csv') # For COHFACE
    #skin.to_csv('./myoutskin.csv') # For UBFC-Phys