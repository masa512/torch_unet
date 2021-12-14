import torch
import torch.nn as nn
import numpy as np

# Class : Double convolution operation with kernel size = 3 for now
# We are using BASELINE unet with no flexibility
class DoubleConv(nn.module):
    # This implementation includes the first two same conv followed by batch norm & Relu activation
    def __init__(self, kernel_size = 3, in_channels:int, out_channels:int):
        super.__init__()
    self.double_conv = nn.Sequential
    (
        nn.Conv2D(in_channels = self.in_channels, out_channels=self.out_channels,kernel_size = self.kernel_size, padding = 'same')
        nn.Batch.Norm2d(num_features = self.out_channels)
        nn.Relu(inplace=True)
        nn.Conv2D(in_channels = self.out_channels, out_channels=self.out_channels,kernel_size = self.kernel_size, padding = 'same')
        nn.Batch.Norm2d(num_features = self.out_channels)
        nn.Relu(inplace=True)
    )
    def forward(self,x):
        return self.double_conv(x)

class DownEncoder(nn.module):
    # This implementation involves Transpose Convolution followed by Double conv
    def __init__(self, kernel_size = 3, in_channels:int, out_channels:int):
        super.__init__()
    self.maxpool_conv = nn.Sequential
    (
        nn.MaxPool2d(kernel_size=2, stride = 2)
        DoubleConv(kernel_size = self.kernel_size, in_channels = self.in_channels, out_channels=self.out_channels)
    )
    def forward(self,x):
        return self.maxpool_conv(x)

class UpDecoder(nn.module):
    # This implementation involves Downsample (Maxpooling) followed by skip connection followed by a DoubleConv implementation
    # Warning : Everything is in same convolution.
    def __init__(self, kernel_size = 3, in_channels:int, out_channels:int):
        super.__init__()
    self.upsample = nn.ConvTranspose2d(kernel_size = 2, in_channels = self.in_channels, out_channels=self.out_channels, stride = 2)
    self.double_conv = DoubleConv(kernel_size = self.kernel_size, in_channels = self.in_channels, out_channels=self.out_channels)
    
    def forward(self, x1, x2=None):
        """
        Remove the last skip connection
        """
        y = self.upsample(x1)
        if x2 is not None:
            y = torch.cat([x1,x2], dim=1)
        return self.double_conv(y)
        
        
class OutConv(nn.module):
    # This implementation just takes care of the 1x1 (point-wise) convolution right before extracting the output feature map
    def __init__(self,in_channels:int, out_channels:int):
        super.__init__()
    self.out_conv = nn.Conv2D(in_channels = self.in_channels, out_channels=self.out_channels,kernel_size = self.kernel_size, padding = 'same')
    def forward(self,x):
        return self.out_conv(x)
