# -*- encoding: utf-8 -*-
"""
@ Author  ：
@ File    : config.py
@ Time    ：2021/5/7 20:36
"""
from model import ConvLstm
from collections import OrderedDict
import torch
import numpy as np

class Config():
    def __init__(self, seq_in, seq_out):
        """
        :param seq_in:
        :param seq_out:
        """
        # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
        self.sequences_in = seq_in
        self.sequences_out = seq_out
        self.b = 1920 
        self.width = 2 * self.b  
        self.height = 2 * self.b  

        self.pic = seq_in 
        self.dtday = 30
        self.percent = 0.7 

        self.batch = 1

        self.num_workers = 0  
        self.num_epochs = 40  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare for earlystopping
        self.patience = 7

        self.half = False

        self.start_mag = 4.0
        self.start_value = 100.0 

        # prepare for weightedMseMae in Shi et al. (2015)
  
        self.sample = "/data/earthquake_zone/sample_" + str(
            self.height) + "_6" + "_collection_mag" + str(int(self.start_mag)) + ".npz"

        self.layerAccess = ([427, 502, 611, 786, 1133], [2706, 3053, 3228, 3337, 3412])
        self.dx = 1
        self.dy = 6 

        self.balancing_weights = [0.000177210579159563, 0.01, 1.0]
        self.THRESHOLDS = np.array([4]) 
        self.normal_loss_global_scale = 1 / 1875

        self.pre_parameters = (0.0003, 20000, 0.7)  # (0.001, 10000, 0.9)

        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 4, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 192, 3, 1, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3)]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [192, 192, 3, 1, 1]}),
                         OrderedDict({'deconv2_leaky_1': [192, 64, 4, 4, 0]}),
                         OrderedDict({'deconv3_relu_1': [64, 8, 7, 5, 1],
                                      'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
    

config = Config(seq_in=9, seq_out=1)
