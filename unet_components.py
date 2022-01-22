import torch
import torch.nn as nn
import numpy as np

# Class : Double convolution operation with kernel size = 3 for now
# We are using BASELINE unet with no flexibility
class DoubleConv(nn.Module):
    # This implementation includes the first two same conv followed by batch norm & Relu activation
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels,kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, out_channels=out_channels,kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace=True) 
        )
    def forward(self,x):
        return self.double_conv(x)
        
class DownEncoder(nn.Module):
    # This implementation involves Transpose Convolution followed by Double conv
    def __init__(self, in_channels:int, out_channels:int, kernel_size = 3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride = 2),
            DoubleConv(kernel_size = kernel_size, in_channels = in_channels, out_channels= out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)

class UpDecoder(nn.Module):
    # This implementation involves Downsample (Maxpooling) followed by skip connection followed by a DoubleConv implementation
    # Warning : Everything is in same convolution.
    def __init__(self, in_channels:int, out_channels:int, kernel_size = 3, mode = 'bicubic'):
        super().__init__()
        if mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(kernel_size = 2, in_channels = in_channels, out_channels= out_channels, stride = 2)

        else:
            self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor = 2, mode = mode),
                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1)
                )
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
        
class UpDecoder_1(nn.Module):
    # This implemention still performs the same way as regular upsample layer except
    # We use upsample by 8 and conv by 4
    def __init__(self, in_channels:int, out_channels:int, kernel_size = 3, mode = 'bilinear', upfactor:int = 4):
        super().__init__()

        # THE UPSCALING PART : Add single layer doing the upscaling by 4
        up_modules = []
        if mode == 'deconv':
            '''
            for i in range(torch.log2(upfactor)):
                up_modules.append(nn.ConvTranspose2d(kernel_size = 2, 
                                                  in_channels = in_channels, 
                                                  out_channels= in_channels, 
                                                  stride = 2
                    ))
            '''
            up_modules.append(nn.ConvTranspose2d(kernel_size = upfactor,
                                                 in_channels = in_channels,
                                                 out_channels = out_channels,
                                                 stride = upfactor
                    ))
        else:
            '''
            for i in range(round(np.log2(upfactor))):
                up_modules.append(nn.Upsample(scale_factor = 2, mode = mode))
            '''
            up_modules.append(nn.Upsample(scale_factor = upfactor, mode = mode))
            up_modules.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1)) 

        # THE DOWNSCALING PART
        down_modules = []
        for i in range(round(np.log2(upfactor))-1):
            down_modules.append(nn.Conv2d(in_channels = out_channels, 
                                          out_channels= out_channels,
                                          kernel_size = kernel_size, 
                                          groups = out_channels,
                                          stride = 2, 
                                          padding = (kernel_size//2,kernel_size//2)))

        self.upsample = nn.Sequential(*up_modules,*down_modules)
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
        x = self.out_conv(x)
        return nn.Sigmoid()(x) # regression