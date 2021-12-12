# helper function/class for the training, mainly
from torch.utils.data import Dataset
from torch.torchvision import transforms
import numpy as np
from pathlib import Path 
from os import path.join as join
from os import listdir
from PIL import Image 
import glob

class unet_dataset(Dataset):
    def __init__(self, indir:str , outdir:str , xmin = 0, xmax = 1, ymin = 0, ymax = 1):
        self.indir = indir
        self.outdir = outdir
        self.xrange = [xmin,xmax]
        self.yrange = [ymin,ymax]
        self.in_names = [splitext(file)[0] for file in listdir(indir) if not file.startswith('.')] # Extract the name before tif of input image as an ID
    def __len__:
        return len(names)
    def __getitem__(self,idx):
        # This method returns dictionary of {in, out} pair given the idx AS TORCH TENSORS
        # The format is -> GT name = Input image just with z(idx) with idx(GT) =/= idx(input)
        # Find index in string to locate "z" and find string that matches substring right upto z
        in_name = join(self.indir,self.in_names[idx]+'.tif')
        out_name = glob.glob(join(self.indir,self.in_names[0:self.in_names[idx].find(s)],'*.tif'))

        # Read images as torch tensor (Might need to fix)
        PIL2tensor = transforms.ToTensor() # Define PIL -> Tensor function 
        im_in = PIL2tensor(Image.open(in_name))
        im_out = PIL2tensor(Image.open(out_name))
        return im_in, im_out
        
    def __rescale__(self,im_in,im_out):
        '''
        This function rescales np image according to x,y range and outputs rescaled input and output image
        '''
        scaled_in = (im_in-self.xmin)/(self.xmax-self.xmin)
        scaled_out = (im_in-self.ymin)/(self.ymax-self.ymin)

        return scaled_in, scaled_out
        

