# Training script for pytorch Unet 
import torch
from pathlib import Path
import os
from torch.utils.data import DataLoader
import numpy as np
# Define the file path for input/output pair

_filepath = ''
indir = os.path.join(_filepath,'inputs')
gtdir = os.path.join(_filepath,'gt')
dir_checkpoint = os.path.join(_filepath,'check_points')

def train_unet(
    device
    num_epochs: int = 1
    batch_size: int = 1
    learning_rate = 1E-4
    r_train = 0.7
):

# Step 1 : Load dataset

dataset = unet_dataset(indir,outdir,xmin = -np.pi, xmax = np.pi, ymin = -np.pi, ymax = np.pi,)

# Step 2 : Load GPU / Network

