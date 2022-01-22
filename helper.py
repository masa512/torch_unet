# helper function/class for the training, mainly
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn 
import torch.nn
from torch.nn.functional import l1_loss as l1_loss
import numpy as np
from pathlib import Path 
from os.path import join as join
from os import listdir
from os.path import splitext
from PIL import Image 
from tifffile import tifffile 
from typing import List
import glob
import pandas as pd
from scipy.signal import medfilt
from torch.fft import fft
import json
import csv

class unet_dataset(Dataset): # For the focus dataset 
    def __init__(self, dir:str , xmin = 0, xmax = 1, ymin = 0, ymax = 1,transform = None, mask_min = 0.4, mask_max = 1):
        self.dir = dir
        self.xrange = [xmin,xmax]
        self.yrange = [ymin,ymax]
        self.transform = transform
        self.dataset_list = pd.read_csv(join(dir,'train_12_19_2021.csv'), header=None).values.tolist()
        self.mask_min = mask_min
        self.mask_max = mask_max

    def __len__(self):
        return len(self.dataset_list)
    def __getitem__(self,idx):
        # This method returns dictionary of {in, out} pair given the idx AS TORCH TENSORS
        # The format is -> GT name = Input image just with z(idx) with idx(GT) =/= idx(input)
        # Find index in string to locate "z" and find string that matches substring right upto z
        in_name = join(self.dir,'Out_Of_Focus/processed',self.dataset_list[idx][0])
        out_name = join(self.dir,'In_Focus',self.dataset_list[idx][1])
        mask_name = join(self.dir,'In_Focus/Mask',self.dataset_list[idx][1])

        # Separate idx and evaluate didx
        z_in = self.dataset_list[idx][0].split("_z")[1].split("_m")[0]
        z_out = self.dataset_list[idx][1].split("_z")[1].split("_m")[0]

        # Read images as torch tensor (Might need to fix)
        #rescaled_in = medfilt(((tifffile.imread(in_name)+np.pi)/(2*np.pi)),kernel_size = 3)
        #rescaled_out = medfilt(((tifffile.imread(out_name)+np.pi)/(2*np.pi)),kernel_size = 3)
        rescaled_in = (tifffile.imread(in_name) + np.pi) / (2 * np.pi)
        rescaled_out = (tifffile.imread(out_name) + np.pi) / (2 * np.pi)
        mask_in = tifffile.imread(mask_name)*(self.mask_max-self.mask_min)+self.mask_min


        im_in = self.transform(rescaled_in.astype('float32'))
        im_out = self.transform(rescaled_out.astype('float32'))
        mask_in = self.transform(mask_in.astype('float32'))

        return {
            'Input' : im_in,
            'GT' : im_out,
            'mask': mask_in,
            'in_name' : self.dataset_list[idx][0],
            'gt_name' : self.dataset_list[idx][1]
        }

class unet_dataset_collagen(Dataset): # For the focus dataset
    def __init__(self, dir:str , xmin = 0, xmax = 1, ymin = 0, ymax = 1,transform = None, mask_min = 0.4, mask_max = 1):
        self.dir = dir
        self.xrange = [xmin,xmax]
        self.yrange = [ymin,ymax]
        self.dataset_list = []
        self.transform = transform
        self.mask_min = mask_min
        self.mask_max = mask_max

    def __len__(self):
        files =
        return
    def __getitem__(self,idx):
        # This method returns dictionary of {in, out} pair given the idx AS TORCH TENSORS
        # The format is -> GT name = Input image just with z(idx) with idx(GT) =/= idx(input)
        # Find index in string to locate "z" and find string that matches substring right upto z
        in_name = join(self.dir,'Out_Of_Focus/processed',self.dataset_list[idx][0])
        out_name = join(self.dir,'In_Focus',self.dataset_list[idx][1])
        mask_name = join(self.dir,'In_Focus/Mask',self.dataset_list[idx][1])

        # Separate idx and evaluate didx
        z_in = self.dataset_list[idx][0].split("_z")[1].split("_m")[0]
        z_out = self.dataset_list[idx][1].split("_z")[1].split("_m")[0]

        # Read images as torch tensor (Might need to fix)
        #rescaled_in = medfilt(((tifffile.imread(in_name)+np.pi)/(2*np.pi)),kernel_size = 3)
        #rescaled_out = medfilt(((tifffile.imread(out_name)+np.pi)/(2*np.pi)),kernel_size = 3)
        rescaled_in = (tifffile.imread(in_name) + np.pi) / (2 * np.pi)
        rescaled_out = (tifffile.imread(out_name) + np.pi) / (2 * np.pi)
        mask_in = tifffile.imread(mask_name)*(self.mask_max-self.mask_min)+self.mask_min


        im_in = self.transform(rescaled_in.astype('float32'))
        im_out = self.transform(rescaled_out.astype('float32'))
        mask_in = self.transform(mask_in.astype('float32'))

        return {
            'Input' : im_in,
            'GT' : im_out,
            'mask': mask_in,
            'in_name' : self.dataset_list[idx][0],
            'gt_name' : self.dataset_list[idx][1]
        }


# Layer loss function

def layer_combined_loss(gt_batch,pred_layer):
    # returns a compressed loss
    conv_compress = nn.Conv2d(in_channels = pred_layer.shape[1], out_channels = 1, kernel_size = 1)(pred_layer)
    gt_compressed = transforms.functional.resize(size = [pred_layer.shape[2],pred_layer.shape[3]]) (gt_batch)
    criterion = nn.MSELoss()
    return criterion(conv_compress,gt_compressed)

# Custom noise transformation

class AddGaussNoise():
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self,tensor): # takes in tensor AddGaussNoise()(tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class SSIM_loss(nn.Module):
    def __init__(self,windowed:bool = False, k1 = 0.01, k2 = 0.03, bit_depth:int = 32):
        super().__init__()
        self.windowed = windowed
        #self.c1 = (k1*(2**bit_depth-1))**2
        #self.c2 = (k2*(2**bit_depth-1))**2
        self.c1 = (k1*(1))**2
        self.c2 = (k2*(1))**2
    def forward(self,y,p):
        if not self.windowed: # For now!
            mup = torch.mean(p)
            muy = torch.mean(y)
            sigy = torch.std(y)
            sigp = torch.std(p)
            sigyp = torch.mean((y-muy)*(p-mup))
            return 1- ((2*mup*muy+self.c1)*(2*sigyp+self.c2))/((mup**2+muy**2+self.c1)*(sigy**2+sigp**2+self.c2))

class SSIM_score(nn.Module):
    # Returns the score for SSIM
    def __init__(self,windowed:bool = False, k1 = 0.01, k2 = 0.03, bit_depth:int = 32, dynamic_range:int = 1):
        super().__init__()
        self.windowed = windowed
        #self.c1 = (k1*(2**bit_depth-1))**2
        #self.c2 = (k2*(2**bit_depth-1))**2
        self.c1 = (k1*(1))**2
        self.c2 = (k2*(1))**2
    def forward(self,y = torch.rand((500,500)),p = torch.rand((500,500))):
        
        mup = torch.mean(p)
        muy = torch.mean(y)
        sigy = torch.std(y)
        sigp = torch.std(p)
        sigyp = torch.mean((y-muy)*(p-mup))

        return ((2*mup*muy+self.c1)*(2*sigyp+self.c2))/((mup**2+muy**2+self.c1)*(sigy**2+sigp**2+self.c2))

class pearson_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        # std x and std y and covariance 
        numerator = torch.sum((x-x.mean())*(y-y.mean()))
        denominator =torch.sqrt(torch.sum((x-x.mean())**2)*torch.sum((y-y.mean())**2))
        return 1-numerator/denominator

class pearson_score(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        # std x and std y and covariance 
        numerator = torch.sum((x-x.mean())*(y-y.mean()))
        denominator =torch.sqrt(torch.sum((x-x.mean())**2)*torch.sum((y-y.mean())**2))
        return numerator/denominator

class PSNR_score(nn.Module):
    def __init__(self, bit_depth:int = 32):
        super().__init__()
        self.MAXI = 1
        self.MSE = nn.MSELoss()
    def forward(self,x,y):
        return 10*torch.log10((1)/self.MSE(x,y)) 
def json_write_wrapper(args,save_path):
    # Args is a dict obj
    json_object = json.dumps(args)
    with open(join(save_path,'train_result.json'),"w") as outfile:
        outfile.write(json_object)

def evaluate_delZ_wrapper(in_name:str = None ,gt_name:str = None, step_size = 0.5):
    z_in = in_name.split("_z")[1].split("_m")[0]
    z_out = gt_name.split("_z")[1].split("_m")[0]
    return str((float(z_in)-float(z_out))*0.5)

def csv_write_wrapper(header:str, entries:str, path:str):
    with open(path,'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        for d in entries:
            csv_writer.writerow([e[1] for e in list(d.items())])

