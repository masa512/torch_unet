import torch
import torchvision
import torch.nn as nn

# what is another name for someone who identifies themnselves as interdisciplinary
# - jack of all trades, masters of none, aka michael.

class LayerLoss(nn.Module):
  def __init__(self, num_inputs: int):
    super().__init__()
    
    self.convs = []
    for i in range(num_inputs):
      self.convs.append(nn.LazyConv2d(1,1,1))
      self.add_module(f'layerLossConv_{i}', self.convs[-1])
     
    mse_loss = nn.MSELoss()
    
  
  def forward(self, target, y, *_input):
    l = 0
    for i, _in in enumerate(_input):
      t = torchvision.transforms.functional.resize(target, _in.shape[2:])
      l = l + mse_loss(self.convs[i](_in), t)
    l = l + mse_loss(y, target)
    return l
