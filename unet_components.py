import torch
import torch.nn as nn
import numpy as np

# Class : Double convolution operation with kernel size = 3 for now
# This implementation has optional squeeze and excite block after each CBR
# We are using BASELINE unet with no flexibility
class DoubleConv(nn.Module):
    # This implementation includes the first two same conv followed by batch norm & Relu activation
    def __init__(self, in_channels, out_channels, kernel_size = 3, do_sqNex:bool = False):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels,kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace=True)
        )
        self.do_sqNex = do_sqNex
        self.sqNex_block = sqNex(out_channels)
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(in_channels = out_channels, out_channels=out_channels,kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace=True) 
        )
    def forward(self,x):
        c1 = self.double_conv1(x)

        # optional sNe
        if self.do_sqNex:
            c1 = self.sqNex_block(c1)
        c2 = self.double_conv2(c1)
        # optional sNe
        if self.do_sqNex:
            c2 = self.sqNex_block(c2)
        return c2

# Squeeze and Excite block
class sqNex(nn.Module):
    # C(important) H W
    # Initializes the squeeze and excite block
    # When input features are fed, the parallel task is done for excitation and original. 
    def __init__(self,n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.qx_branch= nn.Sequential(
            nn.Conv2d(in_channels = self.n_channels ,out_channels = self.n_channels, kernel_size=1),
            nn.Sigmoid() # activation
        )
    def forward(self,x):
        # input features are of size N C H W
        s = self.qx_branch(torch.mean(torch.mean(x,-1,keepdim=True),-2,keepdim = True)) # Returns excitation signal
        # unsqueeze the signal s to have N C H W dimension with all the signals along C axis and channel-wise product
        print(x.shape)
        print(s.shape)
        return x * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(s,-1),-1),0)

class DownEncoder(nn.Module):
    # This implementation involves Transpose Convolution followed by Double conv
    def __init__(self, in_channels:int, out_channels:int, kernel_size = 3, do_sqNex:bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride = 2),
            DoubleConv(kernel_size = kernel_size, in_channels = in_channels, out_channels= out_channels, do_sqNex = do_sqNex)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class UpDecoder(nn.Module):
    # This implementation involves Downsample (Maxpooling) followed by skip connection followed by a DoubleConv implementation
    # Warning : Everything is in same convolution.
    def __init__(self, in_channels:int, out_channels:int, kernel_size = 3):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(kernel_size = 2, in_channels = in_channels, out_channels= out_channels, stride = 2)
        self.double_conv = DoubleConv(kernel_size = kernel_size, in_channels = in_channels, out_channels=out_channels)
    
    def forward(self, x1, x2=None):
        """
        Remove the last skip connection
        """
        
        x1 = self.upsample(x1)
        
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            y = torch.cat([x2,x1], dim=1)
            return self.double_conv(y)
        return x1
        
        
        
class OutConv(nn.Module):
    # This implementation just takes care of the 1x1 (point-wise) convolution right before extracting the output feature map
    def __init__(self,in_channels:int, out_channels:int):
        super(OutConv,self).__init__()
        self.out_conv = nn.Conv2d(in_channels = in_channels, out_channels= out_channels,kernel_size = 1, padding = 'same')
    def forward(self,x):
        return self.out_conv(x)
