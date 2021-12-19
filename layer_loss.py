import torch
import torchvision
import torch.nn as nn
from helper import *

# what is another name for someone who identifies themnselves as interdisciplinary
# - jack of all trades, masters of none, aka michael.

class LayerLoss(nn.Module):
  def __init__(self, num_inputs: int, device: str):
    super().__init__()
    
    self.convs = []
    for i in range(num_inputs):
      self.convs.append(nn.LazyConv2d(1,1,1).to(device))
      self.add_module(f'layerLossConv_{i}', self.convs[-1])
     
    self.mse_loss = nn.MSELoss()
    self.ssim_loss = SSIM_loss()
    self.pcc_loss = pearson_loss()
    
  
  def forward(self, target, y, *_input):
    l = 0
    for i, _in in enumerate(_input):
      t = torchvision.transforms.functional.resize(target, _in.shape[2:])
      __out = self.convs[i](_in)
      l = l + self.mse_loss(__out, t) + self.ssim_loss(__out, t) + self.pcc_loss(__out, t)
    l = l + self.mse_loss(y, target) + self.ssim_loss(y, target) + self.pcc_loss(y, target)
    return l
  
