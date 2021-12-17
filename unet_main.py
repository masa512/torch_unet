import torch
import torch.nn as nn
import numpy as np
import unet_components as blk
import helper
from unet_components import *
from typing import List

<<<<<<< HEAD


class UNet(nn.Module):
=======
from typing import List


# Full implementation of baseline unet using the component

class Unet(nn.Module):
>>>>>>> df94d583b6dd3dfb760b1ce970a91de0ce0571d5
    def __init__(self, 
                 in_channels = 1, 
                 out_channels = 1, 
                 base_num_filter=32, 
<<<<<<< HEAD
                 decoder_probe_points: List[int] = None 
                 ):

        super(UNet, self).__init__()
        #self.decoder_probe_points = decoder_probe_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_num_filter = base_num_filter
=======
                 decoder_probe_points: List[int] = None ):
        super.__init__()
        self.decoder_probe_points = decoder_probe_points
>>>>>>> df94d583b6dd3dfb760b1ce970a91de0ce0571d5
        
        #-------- Input -------#
        self.input_conv = blk.DoubleConv(kernel_size=3, in_channels = self.in_channels, out_channels = self.base_num_filter)

        #------ Encoder ------#
        self.encoder1 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter, out_channels = self.base_num_filter*2)
        self.encoder2 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter*2, out_channels = self.base_num_filter*4)
        self.encoder3 = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter*4, out_channels = self.base_num_filter*8)

        #------ Bottleneck layer ------
        self.bottle_neck = blk.DownEncoder(kernel_size = 3, in_channels = self.base_num_filter*8, out_channels = self.base_num_filter*16)
        
        #------ Decoder -----#
        self.decoder1 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*16, out_channels = self.base_num_filter*8)
        self.decoder2 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*8, out_channels = self.base_num_filter*4)
        self.decoder3 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*4, out_channels = self.base_num_filter*2)
        self.decoder4 = blk.UpDecoder(kernel_size = 3, in_channels = self.base_num_filter*2, out_channels = self.base_num_filter)

<<<<<<< HEAD
        '''
        if decoder_probe_points is not None : 
            assert len(self.decoder_probe_points) == 4, f'Size of decoder probe points must be at most the number of decoder blocks'
            for i in range(4):
                assert -1 < self.decoder_probe_points[i] < 4, f'Expected decoder_probe_points at index {i} to be in the range [0, 3]'\
                f', got {self.decoder_probe_points[i]}'
        '''
=======
        assert len(self.decoder_probe_points) == 4, f'Size of decoder probe points must be at most the number of decoder blocks'
        for i in range(4):
            assert -1 < self.decoder_probe_points[i] < 4, f'Expected decoder_probe_points at index {i} to be in the range [0, 3]'\
            f', got {self.decoder_probe_points[i]}'
>>>>>>> df94d583b6dd3dfb760b1ce970a91de0ce0571d5
        
        # ----- Output -------#
        self.out_conv = blk.OutConv(in_channels = self.base_num_filter, out_channels = self.out_channels)
    
    def forward(self, x):
        x1 = self.input_conv(x)

        e1 = self.encoder1(x1)
<<<<<<< HEAD
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
=======
        e2 = self.encoder1(e1)
        e3 = self.encoder1(e2)
>>>>>>> df94d583b6dd3dfb760b1ce970a91de0ce0571d5
        
        bottle_neck = self.bottle_neck(e3)

        d1 = self.decoder1(bottle_neck, e3)
        d2 = self.decoder2(d1, e2)
        d3 = self.decoder3(d2, e1)
<<<<<<< HEAD
        d4 = self.decoder4(d3,x1)
        
        '''
=======
        d4 = self.decoder4(d3)
        
>>>>>>> df94d583b6dd3dfb760b1ce970a91de0ce0571d5
        decoder_block_outputs = [d1, d2, d3, d4]
        decoder_block_intermediate_outputs = []
        for block in self.decoder_probe_points:
            decoder_block_intermediate_outputs.append(decoder_block_outputs[block])        
<<<<<<< HEAD
        '''

        y = self.out_conv(d4)
        return y#, *decoder_block_intermediate_outputs
=======
        
        y = self.out_conv.forward(d4)
        return y, *decoder_block_intermediate_outputs

>>>>>>> df94d583b6dd3dfb760b1ce970a91de0ce0571d5
