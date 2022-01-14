import torch
import torch.nn as nn
import numpy as np
import unet_components as blk
import helper
from unet_components import *
from typing import List


class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_num_filter=32,
                 decoder_probe_points: List[int] = None,
                 do_sqNex: bool = False
                 ):

        super(UNet, self).__init__()
        self.decoder_probe_points = decoder_probe_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_num_filter = base_num_filter

        # -------- Input -------#
        self.input_conv = blk.DoubleConv(kernel_size=3, 
                                         in_channels=self.in_channels, 
                                         out_channels=self.base_num_filter, 
                                         do_sqNex = do_sqNex)

        # ------ Encoder ------#
        self.encoder1 = blk.DownEncoder(kernel_size=3, 
                                        in_channels=self.base_num_filter,
                                        out_channels=self.base_num_filter * 2,
                                        do_sqNex = do_sqNex
                                        )
        self.encoder2 = blk.DownEncoder(kernel_size=3, 
                                        in_channels=self.base_num_filter * 2,
                                        out_channels=self.base_num_filter * 4,
                                        do_sqNex = do_sqNex
                                        )
        self.encoder3 = blk.DownEncoder(kernel_size=3,
                                        in_channels=self.base_num_filter * 4,
                                        out_channels=self.base_num_filter * 8,
                                        do_sqNex = do_sqNex
                                        )

        # ------ Bottleneck layer ------
        self.bottle_neck = blk.DownEncoder(kernel_size=3,
                                           in_channels=self.base_num_filter * 8,
                                           out_channels=self.base_num_filter * 16,
                                           do_sqNex = do_sqNex
                                           )

        # ------ Decoder -----#
        self.decoder1 = blk.UpDecoder(kernel_size=3, in_channels=self.base_num_filter * 16,
                                      out_channels=self.base_num_filter * 8)
        self.decoder2 = blk.UpDecoder(kernel_size=3, in_channels=self.base_num_filter * 8,
                                      out_channels=self.base_num_filter * 4)
        self.decoder3 = blk.UpDecoder(kernel_size=3, in_channels=self.base_num_filter * 4,
                                      out_channels=self.base_num_filter * 2)
        self.decoder4 = blk.UpDecoder(kernel_size=3, in_channels=self.base_num_filter * 2,
                                      out_channels=self.base_num_filter)

        if decoder_probe_points is not None:
            assert len(
                self.decoder_probe_points) <= 4, f'Size of decoder probe points must be at most the number of decoder blocks'
            for i in range(len(self.decoder_probe_points)):
                assert -1 < self.decoder_probe_points[
                    i] < 4, f'Expected decoder_probe_points at index {i} to be in the range [0, 3]' \
                            f', got {self.decoder_probe_points[i]}'

        # ----- Output -------#
        self.out_conv = blk.OutConv(in_channels=self.base_num_filter, out_channels=self.out_channels)

    def forward(self, x):
        x1 = self.input_conv(x)

        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        bottle_neck = self.bottle_neck(e3)

        d1 = self.decoder1(bottle_neck, e3)
        d2 = self.decoder2(d1, None)  # Removed skip originally e2
        d3 = self.decoder3(d2, None)  # Remove skip originally e1
        d4 = self.decoder4(d3, None)  # Remove skip originally x1

        decoder_block_outputs = [d1, d2, d3, d4]
        decoder_block_intermediate_outputs = []
        for block in self.decoder_probe_points:
            decoder_block_intermediate_outputs.append(decoder_block_outputs[block])

        y = self.out_conv(d4)
        return y, decoder_block_intermediate_outputs
