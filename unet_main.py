import torch
import torch.nn as nn
import numpy as np
import unet_components as blk
import helper

# Full implementation of baseline unet using the component

class Unet(nn.Module):
    def __init__(self,in_channels = 1, out_channels = 1, base_num_filter=32):
        super.__init__()

        #-------- Input -------#
        self.input_conv = blk.DoubleConv(kernel_size=3, in_channels = self.in_channels, out_channels = self.base_num_filter)

        #------ Encoder ------#
        self.encoder1 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter, out_channels = self.base_num_filter*2)
        self.encoder2 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter*2, out_channels = self.base_num_filter*4)
        self.encoder3 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter*4, out_channels = self.base_num_filter*8)
        self.encoder4 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter*8, out_channels = self.base_num_filter*16)

        #------ Decoder -----#
        self.decoder1 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*16, out_channels = self.base_num_filter*8)
        self.decoder2 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*8, out_channels = self.base_num_filter*4)
        self.decoder3 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*4, out_channels = self.base_num_filter*2)
        self.decoder4 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*2, out_channels = self.base_num_filter)

        # ----- Output -------#
        self.out_conv = blk. OutConv(in_channels = self.base_num_filter, out_channels = self.out_channels)
    
    def forward(self,x):
        x1 = self.input_conv.forward(x)

        e1 = self.encoder1.forward(x1)
        e2 = self.encoder1.forward(e1)
        e3 = self.encoder1.forward(e2)
        e4 = self.encoder1.forward(e3)

        d1 = self.decoder1.forward(e4,e3)
        d2 = self.decoder2.forward(d1,e2)
        d3 = self.decoder3.forward(d2,e1)
        d4 = self.decoder4.forward(d3,x1)

        y = self.out_conv.forward(d4)
        return y
    # Below are only for the intermediate loss
    def forward_d3(self,x):
        x1 = self.input_conv.forward(x)

        e1 = self.encoder1.forward(x1)
        e2 = self.encoder1.forward(e1)
        e3 = self.encoder1.forward(e2)
        e4 = self.encoder1.forward(e3)

        d1 = self.decoder1.forward(e4,e3)
        d2 = self.decoder2.forward(d1,e2)
        d3 = self.decoder3.forward(d2,e1)
        return d3

    def forward_d2(self,x):
        x1 = self.input_conv.forward(x)

        e1 = self.encoder1.forward(x1)
        e2 = self.encoder1.forward(e1)
        e3 = self.encoder1.forward(e2)
        e4 = self.encoder1.forward(e3)

        d1 = self.decoder1.forward(e4,e3)
        d2 = self.decoder2.forward(d1,e2)
        return d2
    



