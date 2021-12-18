# helper function/class for the training, mainly
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.nn.functional import l1_loss as l1_loss
import numpy as np
from pathlib import Path 
from os.path import join as join
from os import listdir
from os.path import splitext
from PIL import Image 
from tifffile import tifffile 
import glob

'''
class unet_dataset(Dataset):
    def __init__(self, indir:str , outdir:str , xmin = 0, xmax = 1, ymin = 0, ymax = 1):
        self.indir = indir
        self.outdir = outdir
        self.xrange = [xmin,xmax]
        self.yrange = [ymin,ymax]
        self.in_names = [splitext(file)[0] for file in listdir(indir) if not file.startswith('.')] # Extract the name before tif of input image as an ID
    def __len__(self,idx):
        return len(self.in_names)
    def __getitem__(self,idx):
        # This method returns dictionary of {in, out} pair given the idx AS TORCH TENSORS
        # The format is -> GT name = Input image just with z(idx) with idx(GT) =/= idx(input)
        # Find index in string to locate "z" and find string that matches substring right upto z
        in_name = join(self.indir,self.in_names[idx]+'.tif')
        out_name = glob.glob(join(self.indir,self.in_names[0:self.in_names[idx].find(s)],'*.tif'))

        # Read images as torch tensor (Might need to fix)
        PIL2tensor = transforms.ToTensor() # Define PIL -> Tensor function 
        im_in = PIL2tensor(Image.open(in_name)).contiguous()
        im_out = PIL2tensor(Image.open(out_name)).contiguous()

        scaled_in, scaled_out = self.__rescale__(im_in,im_out)
        return {
            'Input' : scaled_in,
            'GT' : scaled_out
        }
        
    def __rescale__(self,im_in,im_out):

        #This function rescales np image according to x,y range and outputs rescaled input and output image

        scaled_in = (im_in-self.xmin)/(self.xmax-self.xmin)
        scaled_out = (im_in-self.ymin)/(self.ymax-self.ymin)

        return scaled_in, scaled_out
'''

class unet_dataset(Dataset):
    def __init__(self, indir:str , outdir:str , xmin = 0, xmax = 1, ymin = 0, ymax = 1,transform = None):
        self.indir = indir
        self.outdir = outdir
        self.xrange = [xmin,xmax]
        self.yrange = [ymin,ymax]
        self.in_names = [splitext(file)[0] for file in listdir(indir) if not file.startswith('.')] # Extract the name before tif of input image as an ID
        self.transform = transform

    def __len__(self):
        return len(self.in_names)
    def __getitem__(self,idx):
        # This method returns dictionary of {in, out} pair given the idx AS TORCH TENSORS
        # The format is -> GT name = Input image just with z(idx) with idx(GT) =/= idx(input)
        # Find index in string to locate "z" and find string that matches substring right upto z
        in_name = join(self.indir,self.in_names[idx]+'.tif')
        out_name = in_name

        # Read images as torch tensor (Might need to fix)
        '''
        im_in = tifffile.imread(in_name)
        im_out = tifffile.imread(out_name)
        print(im_in, im_out)
        scaled_in, scaled_out = self.__rescale__(im_in,im_out)
        return {
            'Input' : scaled_in,
            'GT' : scaled_out
        }
        '''
        im_in = tifffile.imread(in_name)
        im_out = tifffile.imread(out_name)
        
        return {
            'Input' : self.transform((im_in-self.xrange[0])/(self.xrange[1]-self.xrange[0])),
            'GT' : self.transform((im_out-self.yrange[0])/(self.yrange[1]-self.yrange[0]))
        }


# Layer loss function

def layer_combined_loss(network,gt_batch,pred_batch):
    # returns a compressed loss
    d2_gt = network.forward_d2(gt_batch)
    d2_pred = network.forward_d2(pred_batch)

    d3_gt = network.forward_d3(gt_batch)
    d3_pred = network.forward_d3(pred_batch)

    return l1_loss(d2_gt,d2_pred)+l1_loss(d3_gt,d3_pred)


# Custom noise transformation

class AddGaussNoise():
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self,tensor): # takes in tensor AddGaussNoise()(tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean




