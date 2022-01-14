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

class unet_dataset(Dataset): # For the focus dataset 
    def __init__(self, dir:str , xmin = 0, xmax = 1, ymin = 0, ymax = 1,transform = None):
        self.dir = dir
        self.xrange = [xmin,xmax]
        self.yrange = [ymin,ymax]
        self.transform = transform
        self.dataset_list = pd.read_csv(join(dir,'train_12_19_2021.csv'), header=None).values.tolist()

    def __len__(self):
        return len(self.dataset_list)
    def __getitem__(self,idx):
        # This method returns dictionary of {in, out} pair given the idx AS TORCH TENSORS
        # The format is -> GT name = Input image just with z(idx) with idx(GT) =/= idx(input)
        # Find index in string to locate "z" and find string that matches substring right upto z
        in_name = join(self.dir,'Out_Of_Focus/processed',self.dataset_list[idx][0])
        out_name = join(self.dir,'In_Focus',self.dataset_list[idx][1])

        # Separate idx and evaluate didx
        z_in = self.dataset_list[idx][0].split("_z")[1].split("_m")[0]
        z_out = self.dataset_list[idx][1].split("_z")[1].split("_m")[0]

        # Read images as torch tensor (Might need to fix)
        #rescaled_in = medfilt(((tifffile.imread(in_name)+np.pi)/(2*np.pi)),kernel_size = 3)
        #rescaled_out = medfilt(((tifffile.imread(out_name)+np.pi)/(2*np.pi)),kernel_size = 3)
        rescaled_in = (tifffile.imread(in_name) + np.pi) / (2 * np.pi)
        #rescaled_out = (tifffile.imread(out_name) + np.pi) / (2 * np.pi)
        rescaled_out = (tifffile.imread(out_name) + np.pi) / (2 * np.pi)
        im_in = self.transform(rescaled_in.astype('float32'))
        im_out = self.transform(rescaled_out.astype('float32'))

        # GET FFT and stack as real : 
        fft_in = torch.cat((torch.real(fft(im_in)),torch.imag(fft(im_in))),dim=0)
        return {
            'Input' : im_in,
            'GT' : im_out,
            'FFT_in' : fft_in,
            'delZ' : int(z_in)-int(z_out)
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
        self.c1 = (k1*(2**bit_depth-1))**2
        self.c2 = (k2*(2**bit_depth-1))**2
    def forward(self,y,p):
        if not self.windowed: # For now!
            mup = torch.mean(p)
            muy = torch.mean(y)
            sigy = torch.std(y)
            sigp = torch.std(p)
            sigyp = torch.mean((y-muy)*(p-mup))
            return 1- ((2*mup*muy+self.c1)*(2*sigyp+self.c2))/((mup**2+muy**2+self.c1)*(sigy**2+sigp**2+self.c2))

class pearson_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        # std x and std y and covariance 
        numerator = torch.sum((x-x.mean())*(y-y.mean()))
        denominator =torch.sqrt(torch.sum((x-x.mean())**2)*torch.sum((y-y.mean())**2))
        return 1-numerator/denominator

def json_write_wrapper(args,save_path):
    # Args is a dict obj
    json_object = json.dumps(args, indent = 4)
    with open(join(save_path,'train_result.json'),"w") as outfile:
        outfile.write(json_object)